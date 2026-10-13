"""Portable scoring evidence for frozen initialization reports; never loads checkpoints.

Original report bytes and absolute provenance remain unchanged. A separate
manifest maps original paths to retained local files and explicitly inventories
omitted tensors. Replay verifies those files, independently re-scores retained
predictions, then calls the archived reporter's compute/render entry points.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPORTERS={'dora-init-report-v1':'experiments.init_report',
           'dora-init-confirm-report-v1':'experiments.init_confirm_report',
           'dora-init-extension-report-v1':'experiments.init_extension_report'}
OMITTED={'.pt','.pth','.npy','.npz','.safetensors','.pyc'}
SCOPE='Saved-prediction scoring and report replay only. Tensor payloads are omitted; their original hashes are retained as provenance, not reverified checkpoint contents. No model inference, training or full tensor audit is replayed.'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def local(root,name):
    """Reject traversal, symlink escapes and absolute archive member paths."""
    part=Path(name)
    if part.is_absolute() or not part.parts or any(p in ('.','..') for p in part.parts):raise ValueError('Unsafe portable path')
    target=(root/part).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError('Portable path escapes bundle')
    return target


def verify_file(path,binding):
    if not path.is_file() or path.stat().st_size!=binding['bytes'] or sha(path)!=binding['sha256']:
        raise ValueError(f'Portable file failed byte/hash verification: {path}')


def source_closure(source,entry):
    """Capture actual imports in an isolated process, not guessed import names."""
    script="""import importlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(root));importlib.import_module(sys.argv[2])
rows={}
for name,module in list(sys.modules.items()):
 if name=='experiments' or name.startswith('experiments.'):
  path=pathlib.Path(module.__file__).resolve()
  assert path.is_relative_to(root),(name,path)
  rows[name]=str(path.relative_to(root))
print(json.dumps(rows))
"""
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'-I','-c',script,str(source),entry],cwd=source,env=environment,text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def bundle(report_dir,source_root,output,part_bytes=40_000_000):
    report_dir=Path(report_dir).resolve();source_root=Path(source_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new portable output directory')
    if not isinstance(part_bytes,int) or part_bytes<=0:raise ValueError('part_bytes must be positive')
    original=read(report_dir/'report_manifest.json');payload=read(report_dir/'report_inputs.json')
    entry=REPORTERS[payload['schema']]
    for name,digest in original['outputs'].items():
        path=local(report_dir,name)
        if sha(path)!=digest:raise ValueError(f'Original report changed: {name}')
    if original['outputs'].get('report_inputs.json')!=sha(report_dir/'report_inputs.json'):raise ValueError('Unbound report input')
    closure=source_closure(source_root,entry)
    def bound_digests(value):
        if isinstance(value,dict):return set().union(*(bound_digests(v) for v in value.values())) if value else set()
        if isinstance(value,list):return set().union(*(bound_digests(v) for v in value)) if value else set()
        return {value} if isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value) else set()
    payload_sources=bound_digests(payload)
    snapshot_path=source_root.parent/'execution_snapshot.json'
    snapshot=read(snapshot_path) if snapshot_path.exists() else {}
    for module,name in closure.items():
        digest=sha(source_root/name)
        if module==entry and original.get('source_sha256') not in (None,digest):raise ValueError('Frozen reporter differs from report manifest')
        body=ast.parse((source_root/name).read_text()).body
        empty_init=module=='experiments' and all(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) for n in body)
        if digest not in payload_sources and not empty_init:
            raise ValueError(f'Imported source is not bound to the original report payload: {name}')
    output.mkdir(parents=True);stage=output/'_stage';stage.mkdir()
    files={};mapping={};excluded={}
    def keep(path,name,expected=None):
        path=Path(path);binding={'sha256':sha(path),'bytes':path.stat().st_size}
        if expected and any(binding[key]!=expected[key] for key in ('sha256','bytes')):raise ValueError(f'Input changed: {path}')
        target=local(stage,name);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():verify_file(target,binding)
        else:shutil.copyfile(path,target);verify_file(target,binding)
        if name in files and files[name]!=binding:raise ValueError('Portable path collision')
        files[name]=binding
        return binding
    def evidence(path,expected=None,role='report_evidence'):
        path=Path(path).resolve();key=str(path)
        if key in mapping or key in excluded:return
        if path.suffix.lower() in OMITTED:
            if expected is None:raise ValueError('Excluded supplemental tensor needs an original binding')
            excluded[key]={**expected,'reason':'tensor_or_bytecode_payload_omitted','verification':'Original report binding retained; payload not read by bundler'}
            return
        binding=expected or {'sha256':sha(path),'bytes':path.stat().st_size}
        name='evidence/'+binding['sha256']
        keep(path,name,binding);mapping[key]={**binding,'path':name,'role':role}
    try:
        for name,value in payload['evidence'].items():evidence(name,value)
        # Retain protocols and auditor programs that current Evidence omitted.
        for root_name in sorted({data['root'] for data in payload['tasks'].values()}):
            root=Path(root_name);manifest=read(root/'manifest.json')
            for name in ('reference/protocol.md','protocol.md','reference/reference_provenance.json','controls_audit.json','pilots_audit.json'):
                path=root/name
                if path.exists():
                    if name.endswith('protocol.md') and sha(path)!=manifest['protocol_sha256']:raise ValueError('Supplemental protocol differs')
                    evidence(path,role='supplementary_provenance')
            for path in (root/'audit_source').glob('*.py'):evidence(path,role='saved_auditor_source')
        for module,name in closure.items():keep(source_root/name,'source/'+name)
        if snapshot:evidence(snapshot_path,role='report_source_execution_snapshot')
        expected_outputs={}
        for name in [*original['outputs'],'report_manifest.json']:
            expected_outputs[name]=keep(report_dir/name,'original_report/'+name)
        keep(__file__,'replay.py')
        manifest={'schema':'dora-portable-scoring-v1','scope':SCOPE,'reporter':entry,
            'report_schema':payload['schema'],'report_sha256':sha(report_dir/'report_inputs.json'),
            'original_report_directory':str(report_dir),'source_root':str(source_root),
            'source_binding_policy':'Every imported local source digest is present in original report payload; package __init__ may contain only a docstring. Snapshot receipt is supplemental provenance, not an unbound source fallback.',
            'source_modules':closure,'source_execution_snapshot_sha256':sha(snapshot_path) if snapshot else None,
            'path_map':mapping,'excluded_evidence':excluded,'files':files,'expected_outputs':expected_outputs,
            'replay_environment':{'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in (['numpy','matplotlib','torch'] if 'experiments.init_extension' in closure else ['numpy','matplotlib'])},
                'scope':'Recorded bundle environment; byte-identical output verification is authoritative. Extension reporter imports its bound registry/math modules, which require PyTorch even though no tensor payload is loaded.'},
            'counts':{'included_original_paths':len(mapping),'excluded_original_paths':len(excluded),
                'included_evidence_bytes':sum(v['bytes'] for v in mapping.values()),
                'excluded_evidence_bytes':sum(v['bytes'] for v in excluded.values())}}
        write(stage/'portable_manifest.json',manifest)
        archive=output/'raw_artifacts.tar.gz'
        names=sorted([*files,'portable_manifest.json'])
        with archive.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
            for name in names:
                path=stage/name;info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o444;info.mtime=0
                with path.open('rb') as src:tar.addfile(info,src)
        receipt={'schema':'dora-portable-archive-v1','scope':SCOPE,'archive_path':archive.name,
            'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
            'portable_manifest_sha256':sha(stage/'portable_manifest.json'),'part_bytes':part_bytes,'archive_parts':[],
            'counts':manifest['counts']}
        if archive.stat().st_size>part_bytes:
            with archive.open('rb') as stream:
                while content:=stream.read(part_bytes):
                    path=output/f'raw_artifacts.tar.gz.part-{len(receipt["archive_parts"]):03d}';path.write_bytes(content)
                    receipt['archive_parts'].append({'path':path.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        shutil.copyfile(__file__,output/'replay.py')
        receipt['replay_tool']={'path':'replay.py','sha256':sha(output/'replay.py'),'bytes':(output/'replay.py').stat().st_size}
        write(output/'bundle_manifest.json',receipt)
        packages='numpy matplotlib'+(' torch' if 'experiments.init_extension' in closure else '')
        (output/'README.txt').write_text('Portable scoring evidence\n\n'+SCOPE+'\n\n'
            'Verify bundle_manifest.json and replay.py hashes against a trusted publication receipt.\n'
            'Required Python packages: '+packages+'. Matching recorded versions helps byte-identical plot rendering.\n'
            'Run from this folder: python replay.py replay --bundle-dir . --output ../rebuilt\n'
            'Add --parts-only to force verified ordered-part assembly. Original host inputs and network are denied.\n'
            'The replay_audit.json records raw-prediction rescoring, source imports, exclusions and output hashes.\n'
            'A new empty output path is required. Keep every listed archive part in manifest order.\n')
        return receipt
    finally:shutil.rmtree(stage)


class MappedEvidence:
    def __init__(self,root,manifest):self.root=root;self.manifest=manifest
    def path(self,original):
        name=str(Path(original))
        if name not in self.manifest['path_map']:raise ValueError(f'Required scoring input absent from portable mapping: {name}')
        row=self.manifest['path_map'][name];return local(self.root,row['path'])
    def add(self,original,expected=None):
        path=self.path(original);binding=self.manifest['path_map'][str(Path(original))];verify_file(path,binding)
        if expected is not None and binding['sha256']!=expected:raise ValueError('Mapped original digest differs')
        return binding['sha256']
    def read(self,original,expected=None):self.add(original,expected);return read(self.path(original))
    def lines(self,original,expected=None):self.add(original,expected);return [json.loads(line) for line in self.path(original).read_text().splitlines() if line]


def rescore(payload,primary,evidence,reporter):
    count=0;fits=0;selections=0
    for data in payload['tasks'].values():
        root=Path(data['root']);task=data.get('training_task',data.get('task'))
        manifest=evidence.read(root/'manifest.json')
        if payload['schema']=='dora-init-confirm-report-v1':reporter.verify_registry(evidence,data['registry'])
        else:
            selected=evidence.read(root/'selections.json');primary.selections(manifest['registry'],selected,task)
            if selected['manifest_sha256']!=evidence.add(root/'manifest.json'):raise ValueError('Selection manifest binding differs')
            if data['selections']!=selected['selections']:raise ValueError('Normalized selections differ')
        selections+=1
        dataset='scifact' if data['dataset']=='scifact_transfer' else data['dataset']
        split=evidence.read(root/(dataset+'_splits.json' if task in ('scifact','nfcorpus') else 'split_manifest.json'))
        gold=evidence.read(root/(dataset+'_qrels.json')) if task in ('scifact','nfcorpus') else None
        for fit in data['fits'].values():
            if fit['status']!='pass':continue
            file='receipt.json' if task=='extraction' else 'fit.json' if task=='aircraft' else 'result.json'
            row=evidence.read(Path(fit['fit_id'])/file)
            if task=='extraction':
                row={**row,'result':evidence.read(root/row['artifact_directory']/'result.json',row['result_sha256'])}
            ids,strata,scores,metrics=primary.rescore(evidence,root,task,dataset,row,split,gold)
            if ids!=data['example_ids'] or strata!=data['strata']:raise ValueError('Normalized example IDs/strata differ')
            if len(scores)!=len(fit['scores']):raise ValueError('Normalized score count differs')
            for a,b in zip(scores,fit['scores']):primary.close(a,b,1e-12)
            for key,value in metrics.items():primary.close(value,fit['metrics'][key],3e-7 if task=='aircraft' else 1e-12)
            count+=len(scores);fits+=1
    return {'prediction_rows_rescored':count,'dataset_fits_rescored':fits,'validation_selections_checked':selections}


def worker(root,output,audit):
    root=Path(root).resolve();output=Path(output).resolve();manifest=read(root/'portable_manifest.json')
    if output.exists():raise ValueError('Use a new replay output directory')
    for name,binding in manifest['files'].items():verify_file(local(root,name),binding)
    payload=read(root/'original_report/report_inputs.json')
    if sha(root/'original_report/report_inputs.json')!=manifest['report_sha256']:raise ValueError('Original input digest differs')
    if REPORTERS.get(payload['schema'])!=manifest['reporter']:raise ValueError('Reporter schema mismatch')
    for original,binding in payload['evidence'].items():
        present=manifest['path_map'].get(original);absent=manifest['excluded_evidence'].get(original)
        if (present is None)==(absent is None):raise ValueError('Evidence must be either retained or explicitly omitted')
        declared=present or absent
        if any(declared[k]!=binding[k] for k in ('sha256','bytes')):raise ValueError('Original evidence binding changed')
        if absent and Path(original).suffix.lower() not in OMITTED:raise ValueError('Required nontensor evidence omitted')
    denied={Path(p).resolve() for p in {*manifest['path_map'],*manifest['excluded_evidence']}}
    denied_roots={Path(d['root']).resolve() for d in payload['tasks'].values()}
    denied_roots.update(Path(manifest[k]).resolve() for k in ('original_report_directory','source_root'))
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path in denied or any(path.is_relative_to(parent) for parent in denied_roots):raise RuntimeError('Replay attempted original host evidence access')
        if event in ('socket.connect','socket.getaddrinfo'):raise RuntimeError('Portable replay attempted network access')
    sys.addaudithook(guard)
    sys.path.insert(0,str(root/'source'))
    reporter=importlib.import_module(manifest['reporter']);primary=importlib.import_module('experiments.init_report')
    def verify_imports():
        loaded={}
        for name,module in list(sys.modules.items()):
            if name=='experiments' or name.startswith('experiments.'):
                if name not in manifest['source_modules']:raise ValueError('Unarchived local module imported')
                wanted=local(root,'source/'+manifest['source_modules'][name])
                if Path(module.__file__).resolve()!=wanted:raise ValueError('Local module imported outside frozen archive')
                loaded[name]=str(wanted.relative_to(root))
        if set(loaded)!=set(manifest['source_modules']):raise ValueError('Frozen imported module closure differs')
        return loaded
    verify_imports();scoring=rescore(payload,primary,MappedEvidence(root,manifest),reporter)
    result=reporter.compute(payload) if payload['schema']=='dora-init-confirm-report-v1' else reporter.compute(payload,payload['bootstrap_replicates'])
    reporter.render(payload,result,output);loaded=verify_imports()
    actual={p.name for p in output.iterdir() if p.is_file()}
    if actual!=set(manifest['expected_outputs']):raise ValueError('Rebuilt output file inventory differs')
    for name,binding in manifest['expected_outputs'].items():verify_file(output/name,binding)
    write(audit,{'status':'pass','scope':SCOPE,**scoring,
        'environment':{'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in manifest['replay_environment']['packages']}},'byte_identical_outputs':manifest['expected_outputs'],
        'imported_frozen_modules':loaded,'included_original_paths':len(manifest['path_map']),
        'excluded_tensor_bindings':len(manifest['excluded_evidence']),'excluded_tensor_payloads_verified':0,
        'original_host_paths_denied':True,'network_denied':True,'gpu_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'portable_manifest_sha256':sha(root/'portable_manifest.json')})


def replay(bundle_dir,output,work_parent=None,parts_only=False):
    bundle_dir=Path(bundle_dir).resolve();output=Path(output).resolve();receipt=read(bundle_dir/'bundle_manifest.json')
    if receipt['schema']!='dora-portable-archive-v1':raise ValueError('Unknown archive schema')
    work=Path(tempfile.mkdtemp(prefix='dora_portable_replay_',dir=work_parent))
    archive=local(bundle_dir,receipt['archive_path'])
    if parts_only or not archive.exists():
        archive=work/'raw_artifacts.tar.gz';parts=receipt['archive_parts']
        if not parts:raise ValueError('No archive parts supplied')
        with archive.open('wb') as destination:
            for index,row in enumerate(parts):
                if row['path']!=f'raw_artifacts.tar.gz.part-{index:03d}':raise ValueError('Invalid archive part order')
                source=local(bundle_dir,row['path']);verify_file(source,row)
                with source.open('rb') as stream:shutil.copyfileobj(stream,destination)
    verify_file(archive,{'bytes':receipt['archive_bytes'],'sha256':receipt['archive_sha256']})
    raw=work/'raw';raw.mkdir()
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(set(names))!=len(names):raise ValueError('Duplicate archive member')
        for member in members:
            local(raw,member.name)
            if not member.isfile():raise ValueError('Only regular evidence files are permitted')
        tar.extractall(raw,filter='data')
    if sha(raw/'portable_manifest.json')!=receipt['portable_manifest_sha256']:raise ValueError('Portable manifest differs')
    manifest=read(raw/'portable_manifest.json')
    if set(names)!={*manifest['files'],'portable_manifest.json'}:raise ValueError('Archive file inventory differs')
    for name,binding in manifest['files'].items():verify_file(local(raw,name),binding)
    for path in raw.rglob('*'):
        if path.is_file():path.chmod(0o444)
    for path in sorted((p for p in raw.rglob('*') if p.is_dir()),reverse=True):path.chmod(0o555)
    raw.chmod(0o555)
    audit=work/'replay_audit.json'
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_HOME':str(work/'absent_model_cache'),'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(work/'mpl')}
    command=[sys.executable,'-I',str(raw/'replay.py'),'_worker','--root',str(raw),'--output',str(output),'--audit',str(audit)]
    with (work/'replay.log').open('w') as log:result=subprocess.run(command,cwd=work,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Portable replay failed: {work}/replay.log')
    result=read(audit);result.update(work_directory=str(work),archive_sha256=sha(archive),parts_only=parts_only,
        read_only_archive_files_verified_after_replay=len(manifest['files']))
    for name,binding in manifest['files'].items():verify_file(local(raw,name),binding)
    write(bundle_dir/'replay_audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('bundle');build.add_argument('--report-dir',type=Path,required=True);build.add_argument('--source-root',type=Path,required=True)
    build.add_argument('--output',type=Path,required=True);build.add_argument('--part-bytes',type=int,default=40_000_000)
    check=sub.add_parser('replay');check.add_argument('--bundle-dir',type=Path,required=True);check.add_argument('--output',type=Path,required=True)
    check.add_argument('--work-parent',type=Path);check.add_argument('--parts-only',action='store_true')
    internal=sub.add_parser('_worker');internal.add_argument('--root',type=Path,required=True);internal.add_argument('--output',type=Path,required=True);internal.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args();values=vars(args);command=values.pop('command')
    result={'bundle':bundle,'replay':replay,'_worker':worker}[command](**values)
    print(json.dumps(result if command=='bundle' else {'status':'pass','command':command},indent=2))


if __name__=='__main__':main()
