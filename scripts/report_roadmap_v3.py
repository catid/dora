#!/usr/bin/env python3
"""Current scope and a provisional ETA, grounded in completed diagnostic runs.

Timing extrapolations are arithmetic, not a calibrated four-GPU queue forecast.
This report never treats engineering checks as eligible research trials.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    prior_path=Path('artifacts/roadmap_v3.json')
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
    now=datetime.now(timezone.utc).isoformat()
    anchor=prior.get('first_eta_snapshot_at',prior.get('updated_at',now))
    plan = json.loads(Path('artifacts/run_plan_v3.json').read_text())
    root = Path('/var/tmp/dora-adaptive-rank/runs/v3_full_horizon_checks_20261015')
    timing = []
    for path in sorted(root.glob('*/metadata.json')):
        raw = json.loads(path.read_text())
        if raw['status'] != 'complete':
            continue
        timing.append({key: raw[key] for key in [
            'method', 'updates_finished', 'input_tokens', 'charged_gpu_seconds',
            'setup_s', 'evaluation_s', 'backbone_unchanged', 'budget_delta']}
            | {'metadata': str(path), 'selection_eligible': False})
    stage_b = [r for r in plan['runs'] if r['stage'] in {'B_hpo', 'B_tuning', 'B_screen'}]
    updates = sum(r['updates'] for r in stage_b)
    rates = [r['charged_gpu_seconds'] / r['updates_finished'] for r in timing]
    gpu_hours = [updates * rate / 3600 for rate in rates]
    ideal_hours = [hours / 4 for hours in gpu_hours]
    pipelines = []
    for path in sorted(Path('/var/tmp/dora-adaptive-rank/runs/v3_pipeline_smoke_20261015').glob('*/metadata.json')):
        raw = json.loads(path.read_text())
        pipelines.append({'run_id': path.parent.name, 'method': raw['method'], 'status': raw['status'],
                          'metadata': str(path), 'failure_reason': raw.get('failure_reason'),
                          'selection_eligible': False})
    estimate = {
        'schema': 'rankstudy-v3-roadmap', 'updated_at': now, 'first_eta_snapshot_at': anchor,
        'scope': 'All required stages retained; no reduced-scope completion claim.',
        'actual_hardware': '4 RTX PRO 6000 Blackwell Max-Q GPUs',
        'full_horizon_timing_evidence': timing,
        'pipeline_engineering_checks': pipelines,
        'stage_B_mandatory_training_jobs': len(stage_b), 'stage_B_total_updates': updates,
        'stage_B_training_only_extrapolation_gpu_hours': [min(gpu_hours), max(gpu_hours)],
        'stage_B_training_only_ideal_four_gpu_hours': [min(ideal_hours), max(ideal_hours)],
        'extrapolation_excludes': ['generation evaluation', 'model/data setup', 'conditional rescue trials',
                                  'failed attempts', 'method cost differences', 'scheduler contention',
                                  'remaining engineering and validation'],
        'provisional_elapsed_time_from_this_snapshot': [
            {'milestone': 'Remaining Stage A checks and corrected queue', 'range': '24-48 hours'},
            {'milestone': 'Stage B: all 41 candidates with independent screening seeds', 'range': '2-4 days'},
            {'milestone': 'Stages C/D: fresh-seed, rank-four, export and equal-compute comparisons', 'range': '4-7 days'},
            {'milestone': 'Stage E plus official evaluations and final audit/report', 'range': '1-3 weeks'},
        ],
        'uncertainty': 'Planning estimates, not guarantees. Assume continued four-GPU access. '
                       '7B and generation throughput are not yet calibrated; correctness failures can extend these ranges.',
    }
    Path('artifacts/roadmap_v3.json').write_text(json.dumps(estimate, indent=2) + '\n')
    lines = [
        '# Roadmap and provisional ETA', '', f"Updated {estimate['updated_at']}.", '',
        f"Planning ranges remain anchored to the initial ETA snapshot, {anchor}; refreshing this report does not restart the clock.", '',
        'The study remains in correctness validation. No corrected primary HPO, screening, '
        'confirmation or official benchmark comparison is complete. The 166 quarantined v2 '
        'trials remain excluded. All 41 candidates and the complete requested scope remain registered.', '',
        '| Completed full-horizon engineering run | Updates | Input tokens | Charged training minutes |',
        '|---|---:|---:|---:|',
    ]
    for row in timing:
        lines.append(f"| {row['method']} | {row['updates_finished']} | {row['input_tokens']:,} | {row['charged_gpu_seconds']/60:.2f} |")
    lines += ['', 'These runs used a common untuned configuration and only 16 development groups. '
              'They establish execution and timing, not a method ranking.', '',
              '| Milestone | Provisional cumulative ETA from initial snapshot |', '|---|---|']
    for row in estimate['provisional_elapsed_time_from_this_snapshot']:
        lines.append(f"| {row['milestone']} | {row['range']} |")
    lines += ['',
        'Stage A has now executed all 41 GPU smoke cells and 984 retained synthetic budgeted runs. '
        'Pipeline resume, common export allocation and same-checkpoint DDP/rebuild checks have receipts. '
        'Remaining work includes parser/generation validation, shadow deletion/curvature/gauge diagnostics '
        'and restoration, complete primary instrumentation and the balanced v3 tuning queue. '
        'Strict independent DDP trajectory and four BF16 refactor-parity failures remain disclosed; '
        'five FP32 model export oracles pass. See the validation report for the scope of each check.', '',
        f"Stage B contains **{len(stage_b)} mandatory training jobs**: 492 short searches, "
        '164 full-horizon tuning runs and 123 independent screening runs. Conditional rescue trials are extra. '
        f"At the two measured native rates, training alone extrapolates to **{min(gpu_hours):.1f}–{max(gpu_hours):.1f} GPU-hours**, "
        f"or **{min(ideal_hours):.1f}–{max(ideal_hours):.1f} hours** with ideal four-GPU parallelism. "
        'This arithmetic excludes evaluation, setup, pipeline overhead differences, rescue trials and remaining engineering. '
        'The timing runs used two concurrent GPUs; four-job queue throughput still needs calibration.', '',
        estimate['uncertainty'], '',
        'The 3,064-entry plan includes diagnostics, conditional branches and separate evaluation jobs; '
        'it is not a count of unconditional training runs. See `artifacts/coverage_v3.json` for exact coverage '
        'and `reports/progress.md` for validation evidence.', '',
        'Regenerate: `.venv/bin/python scripts/report_roadmap_v3.py`.',
    ]
    Path('reports/roadmap_v3.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: estimate[k] for k in ['stage_B_mandatory_training_jobs',
                                              'stage_B_training_only_ideal_four_gpu_hours']}))


if __name__ == '__main__':
    main()
