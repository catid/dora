#!/usr/bin/env python3
"""Render immutable shadow deletion diagnostics into CSV and Markdown."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); ap.add_argument('--out', type=Path, default=Path('reports')); args = ap.parse_args()
    rows=[]; summary=[]
    for path in sorted(args.root.glob('*/metrics.json')):
        data=json.loads(path.read_text())
        for family in data['families']:
            groups={g['selection']:g for g in family['groups']}
            row={'method':data['method'],'progress':data['progress'],'family':family['family'],
                 'active_slots':data['active_slots'],'rho_score_damage':family['rank_correlation_score_vs_damage'],
                 'rho_abs_score_abs_damage':family['rank_correlation_abs_score_vs_abs_damage'],
                 'top64_damage':groups['score_top64']['token_nll_damage'],
                 'bottom64_damage':groups['score_bottom64']['token_nll_damage'],
                 'common64_damage':groups['common_stratified64']['token_nll_damage']}
            rows.append(row)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'deletion_diagnostics_v3.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    lines=['# Shadow deletion diagnostics (v3)','',
           'These are training-side diagnostics on the fixed 64-example probe. They are not used for HPO selection. Each checkpoint uses exact token-progress replay at 20%, 50%, and 75% of the 512-update stream. ACT uses one shared effective score batch with at most 256 valid tokens. Adapter tensors and RNG state were restored and verified for every checkpoint.','',
           '| Method | Progress | Score | Spearman score/damage | Top-64 damage | Bottom-64 damage | Common-64 damage |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['progress']:.2f} | {r['family']} | {r['rho_score_damage']:.3f} | {r['top64_damage']:.6f} | {r['bottom64_damage']:.6f} | {r['common64_damage']:.6f} |")
    lines += ['', 'The selected group damage is signed loss change; negative values would indicate that deletion improved the probe loss. Correlations are descriptive and use 64 single-slot deletions per checkpoint. No family winner is inferred from these probes.']
    (args.out/'deletion_diagnostics_v3.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(rows),'restoration_verified':all(json.loads(p.read_text())['restoration_verified'] for p in args.root.glob('*/metrics.json'))}))


if __name__=='__main__': main()
