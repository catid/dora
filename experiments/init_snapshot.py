"""Freeze local source plus experiment inputs and verify import isolation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments import init_common as common


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','gate','registry-root','protocol'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();repo=Path.cwd().resolve();source=output/'source_run'
    assert not (output/'execution_snapshot.json').exists(), 'Use a new snapshot destination'
    gate=common.read(args.gate)
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    for name,digest in gate['source_sha256'].items():
        assert common.sha(repo/name)==digest, name
    source.mkdir(parents=True,exist_ok=False)
    names=sorted({*repo.glob('*.py'),repo/'README.md',
                  *(p for p in (repo/'experiments').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json'))})
    bindings={}
    for path in names:
        name=path.relative_to(repo)
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
        bindings[str(name)]=common.sha(path)
        assert common.sha(destination)==bindings[str(name)]
    inputs=output/'inputs';inputs.mkdir()
    shutil.copyfile(args.gate,inputs/'backend_gate.json')
    shutil.copyfile(args.protocol,inputs/'protocol.md')
    shutil.copytree(args.registry_root,inputs/'registries')
    for path in (inputs/'registries').rglob('*.json'):
        common.load_registry(path,inputs/'protocol.md')
    modules=['experiments.init_common','experiments.init_ablations','experiments.init_data',
             'experiments.init_calibration','experiments.init_loftq','experiments.init_warmstart',
             'experiments.init_retrieval','experiments.init_aircraft','experiments.init_extraction']
    script="import importlib,json,sys; print(json.dumps({n:importlib.import_module(n).__file__ for n in sys.argv[1:]}))"
    environment={**os.environ,'PYTHONPATH':str(source),'CUDA_VISIBLE_DEVICES':''}
    imported=json.loads(subprocess.check_output([sys.executable,'-c',script,*modules],cwd=source,env=environment,text=True))
    assert all(Path(path).resolve().is_relative_to(source) for path in imported.values()), imported
    public=common.sha(source/'dora.py')
    assert public=='5f11aa1c64b6f02fb6bb8fb4558c45b12e1501f4543c40afb654bfc5b6958f24'
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (output/'working_tree_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    (output/'tracked_diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD']))
    manifest=dict(schema='dora-init-execution-snapshot-v1',original_repository=str(repo),source=str(source),
                  git_head=git_head,public_source_sha256=public,source_sha256=bindings,
                  active_module_bindings=imported,
                  inputs_sha256={str(p.relative_to(output)):common.sha(p) for p in inputs.rglob('*') if p.is_file()},
                  tracked_diff_sha256=common.sha(output/'tracked_diff.patch'),
                  scope='Actual task subprocesses must run with this cwd and PYTHONPATH; no symlinks to mutable study source.')
    common.write_once(output/'execution_snapshot.json',manifest)
    print(json.dumps({'source_files':len(bindings),'snapshot':str(output),'module_bindings_verified':len(imported)}))


if __name__=='__main__':
    main()
