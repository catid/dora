"""Require full-budget endpoint tensors and saved predictions to reproduce."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(root, lora=False):
    results=json.loads((root/'results.json').read_text())
    assert results['complete']
    task=results['task']
    if lora:
        references={'lora_conventional': Path('/var/tmp/dora-bench/runs/retrieval')}
    else:
        references={'ab_original':Path('/var/tmp/dora-bench/original_bench')/task,
                    'ab_revised':Path('/var/tmp/dora-bench/runs/retrieval' if task=='scifact' else '/var/tmp/dora-bench/round2/retrieval')}
    rows=[]
    for method, reference_root in references.items():
        prefix='lora' if lora else 'dora_original' if method=='ab_original' else 'dora'
        for seed in (42,43,44):
            reference=reference_root/f'{prefix}{"_r8" if task=="nfcorpus" else ""}_seed{seed}'
            current=next(row for row in results['runs'] if row['method']==method and row['seed']==seed and row['lane']=='tuned')
            directory=root/current['fit_id']
            expected=torch.load(reference/'adapter.pt',map_location='cpu',weights_only=True)
            actual=torch.load(directory/'adapter.pt',map_location='cpu',weights_only=True)
            def renamed(name):
                if method!='ab_original':return name
                return name[:-6]+('lora_B' if name.endswith('lora_A') else 'lora_A') if name.endswith(('lora_A','lora_B')) else name
            assert {renamed(name) for name in expected}==set(actual)
            for name,tensor in expected.items():
                torch.testing.assert_close(actual[renamed(name)],tensor,rtol=0,atol=0)
            predictions={}
            for dataset in ('nfcorpus','scifact') if task=='nfcorpus' else ('scifact',):
                old_predictions=reference/(dataset if task=='nfcorpus' else '')/'test_per_query.json'
                new_predictions=directory/dataset/'test_per_query.json'
                assert old_predictions.read_bytes()==new_predictions.read_bytes(),(method,seed,dataset)
                predictions[dataset]={'reference_sha256':sha(old_predictions),'current_sha256':sha(new_predictions)}
            rows.append({'method':method,'seed':seed,'reference':str(reference),
                         'adapter_tensor_count':len(actual),'adapter_tensors_bitwise_equal':True,
                         'test_rankings_scores_metrics_byte_identical':True,
                         'current_checkpoint_sha256':sha(directory/'adapter.pt'),
                         'reference_checkpoint_sha256':sha(reference/'adapter.pt'),
                         'prediction_sha256':predictions})
    return {'status':'pass','source_sha256':sha(__file__),
            'scope':'Complete full-budget tuned endpoint parity against previously measured runs. Original factor names swapped only.',
            'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--lora',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=check(args.root,args.lora)
    (args.root/'endpoint_quality_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    source=args.root/'audit_source'/Path(__file__).name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':'pass','endpoint_fits':len(result['rows']),'tensors':sum(r['adapter_tensor_count'] for r in result['rows'])}))
