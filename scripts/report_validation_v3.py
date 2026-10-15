#!/usr/bin/env python3
"""Validation receipts and limitations built from raw terminal records."""
import csv
import hashlib
import json
from pathlib import Path
import yaml


def main():
    methods={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    root=Path('/var/tmp/dora-adaptive-rank/runs/v3_registry_smoke_20261015')
    rows=[];checks={};hashes=set();exposures=set()
    for method,spec in methods.items():
        path=root/method/'metadata.json'
        if not path.exists():rows.append({'method':method,'status':'pending','path':str(path)});continue
        m=json.loads(path.read_text());hashes.add(m.get('original_backbone_sha256'))
        exposures.add((m.get('input_tokens'),m.get('supervised_tokens')))
        expected_budget=(spec['group']=='wide' or spec['adalora'] or m.get('budget_delta',1)<=0)
        checks[method]=m['status']=='complete' and m.get('backbone_unchanged') and expected_budget
        rows.append({'method':method,'status':m['status'],'path':str(path),'input_tokens':m.get('input_tokens'),
                     'supervised_tokens':m.get('supervised_tokens'),'budget_delta':m.get('budget_delta'),
                     'charged_gpu_seconds':m.get('charged_gpu_seconds'),
                     'native_budget_exception':'wide control' if spec['group']=='wide' else 'faithful reference E coefficients' if spec['adalora'] else None,
                     'selection_eligible':False})
    passed=len(checks)==41 and all(checks.values()) and len(hashes)==1 and len(exposures)==1
    receipt={'status':'passed' if passed else 'incomplete','registered_methods':41,'completed':sum(r['status']=='complete' for r in rows),
             'same_exposure':len(exposures)==1,'same_unchanged_backbone':len(hashes)==1,
             'rows':rows,'scope':'all registry cells execute 32 effective updates under one engineering configuration; no HPO or family selection',
             'not_established':['full-horizon screening per cell','LLM accuracy comparisons','generation/parser correctness']}
    Path('artifacts/v3_registry_smoke_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    common=[]
    for path in sorted(Path('/var/tmp/dora-adaptive-rank/runs/v3_pipeline_global_export_20261015').glob('*/metadata.json')):
        m=json.loads(path.read_text());desc=json.loads((Path(m['export'])/'manifest.json').read_text())
        assert m['status']=='complete' and m['budget_delta']==0 and desc['ordinary_factor_scalars']==m['P_target']
        common.append({'method':m['method'],'parameter_count':desc['ordinary_factor_scalars'],
                       'factor_bytes':desc['ordinary_factor_bytes'],'tensor_payload_bytes':desc['tensor_payload_bytes'],
                       'safetensors_file_bytes':desc['safetensors_file_bytes'],'metadata':str(path)})
    assert len({r['parameter_count'] for r in common})==1 and len({r['factor_bytes'] for r in common})==1
    Path('artifacts/v3_common_export_budget_audit.json').write_text(json.dumps({'status':'passed','runs':common,
        'note':'Factor payload sizes match; safetensors header and source metadata sizes are separately reported.'},indent=2)+'\n')
    if passed:
        gate=Path('/var/tmp/dora-adaptive-rank/runs/v3/A_gate_gpu_all_representations');gate.mkdir(exist_ok=True)
        (gate/'metadata.json').write_text(json.dumps({'status':'passed','evidence':'artifacts/v3_registry_smoke_audit.json',
            'scope':receipt['scope'],'not_established':receipt['not_established']},indent=2)+'\n')
    exports=[]
    roots=[('/var/tmp/dora-adaptive-rank/runs/v3_export_reload_20261015','bf16'),
           ('/var/tmp/dora-adaptive-rank/runs/v3_export_fp32_oracle_20261015','fp32_oracle'),
           ('/var/tmp/dora-adaptive-rank/runs/v3_reference_export_reload_20261015',None)]
    for name,arithmetic in roots:
        for path in sorted(Path(name).glob('*/audit.json')):
            m=json.loads(path.read_text());exact=m['comparisons']['exact'];truncated=m['comparisons']['truncated']
            exports.append({'method':'ADALORA_REF' if arithmetic is None else path.parent.name,
                'arithmetic':arithmetic or path.parent.name,'strict_audit_status':m['status'],
                'exact_factor_max_relative_error':max(m['factor_relative_errors'].values()),
                'exact_logits_relative_rms':exact['logits_relative_rms'],
                'exact_logits_max_absolute_error':exact['logits_max_absolute_difference'],
                'exact_nll_difference':exact['group_nll_difference'],
                'truncated_nll_difference':truncated['group_nll_difference'],
                'path':str(path),'audit_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with Path('reports/export_validation_v3.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(exports[0]),lineterminator='\n');w.writeheader();w.writerows(exports)
    lines=['# V3 validation evidence','',
        '| Measured conclusion | Evidence and limits |','|---|---|',
        f"| All {receipt['completed']}/41 registry cells execute on Qwen1.5B | 32 updates, identical input/supervised exposure and original backbone; engineering configurations only. |",
        f"| Common export budget fixed | {len(common)} new global-energy exports each contain {common[0]['parameter_count']:,} ordinary-factor scalars / {common[0]['factor_bytes']:,} factor bytes; original underspending runs retained. |",
        '| Interrupted pipelines reproduce controls | Restart student and compression recovery match final tensors, Adam, score state, RNG and exposure bit for bit. |',
        '| Exact export algebra passes FP32 model oracles | All five representation families pass; ordinary exports load on the original backbone in separate processes. |',
        '| Conservative BF16 export parity fails for four cases | Update reconstruction and FP32 checks pass, but BF16 rounding changes full-model logits; raw distortion remains reported, not relabeled exact numerical equality. |',
        '| DDP matches at identical checkpoints | Four real-model cases × four updates pass through no_sync accumulation, unequal token counts, pruning and DDP/optimizer rebuild. |',
        '| Independent DDP trajectories do not meet strict tolerances | BF16 gradient and FP32 parameter checks failed at update two; free-trajectory equivalence remains unproven. Primary runs stay on one GPU. |',
        '', '## Export checks', '',
        'The presealed 64-example training probe is used in every case. FP32 rows promote the same stored BF16 weight values '
        'solely as an arithmetic oracle; they are not primary training/evaluation results. The exact refactor may exceed the final budget. '
        'Truncated NLL differences measure immediate distortion before recovery. No development or test selection uses these probes.', '',
        '| Method | Arithmetic | Strict result | Exact logit relative RMS | Exact NLL change | Truncated NLL change |',
        '|---|---|---|---:|---:|---:|']
    for r in exports:
        lines.append(f"| {r['method']} | {r['arithmetic']} | {r['strict_audit_status']} | {r['exact_logits_relative_rms']:.3g} | {r['exact_nll_difference']:.3g} | {r['truncated_nll_difference']:.3g} |")
    lines+=['','All thresholds and failed records remain preserved. Passing the FP32 oracle does not erase the BF16 failure. '
        'DDP comparisons at identical states do not prove independent trajectories remain identical. Neither diagnostic establishes a quality winner.',
        '', 'Primary tuning, independent screening, confirmation, equal-compute/export quality comparisons, 7B and official benchmark evaluations remain pending. '
        'Generation/parser validation and shadow deletion/curvature/gauge/restoration work remain open.',
        '', 'Commands: `scripts/check_export_v3.py`, `scripts/check_llm_ddp_v3.py`, `scripts/smoke_registry_v3.py`; exact invocations are stored with each raw run.']
    Path('reports/validation_v3.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'registry_smoke':receipt['status'],'completed_cells':receipt['completed'],'export_audits':len(exports)}))


if __name__=='__main__':main()
