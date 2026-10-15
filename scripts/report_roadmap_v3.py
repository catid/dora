#!/usr/bin/env python3
"""Current scope and a provisional ETA, grounded in completed diagnostic runs.

Timing extrapolations are arithmetic, not a calibrated four-GPU queue forecast.
This report never treats engineering checks as eligible research trials.
"""
import json
import time
from collections import Counter
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

    # Once the queue is live, derive status and ETA from its immutable
    # launcher records rather than leaving the original pre-launch wording in
    # the report.  This is an observed queue forecast, not a completion claim.
    hpo_root = Path('/var/tmp/dora-adaptive-rank/runs/v3_hpo_20261015_retry')
    hpo_rows = []
    for path in sorted(hpo_root.glob('*/launcher_status.json')):
        try:
            row = json.loads(path.read_text())
            row['_mtime'] = path.stat().st_mtime
            hpo_rows.append(row)
        except (OSError, json.JSONDecodeError):
            continue
    hpo_counts = dict(Counter(row.get('status', 'unknown') for row in hpo_rows))
    completed_times = sorted(row['_mtime'] for row in hpo_rows if row.get('status') == 'complete')
    hpo_rate = hpo_remaining = hpo_eta = None
    if len(completed_times) >= 2 and completed_times[-1] > completed_times[0]:
        hpo_rate = (len(completed_times) - 1) / ((completed_times[-1] - completed_times[0]) / 60.0)
        remaining = max(0, 492 - len(completed_times))
        hpo_remaining = remaining / (hpo_rate * 60.0) if hpo_rate else None
        if hpo_remaining is not None:
            hpo_eta = datetime.fromtimestamp(time.time() + hpo_remaining * 3600, timezone.utc).isoformat()
    def queue_status(root, expected):
        """Read launcher receipts and derive a cautious observed-rate ETA."""
        rows = []
        root = Path(root)
        paths = root.glob('*/launcher_status.json') if root.exists() else []
        for path in paths:
            try:
                row = json.loads(path.read_text())
                row['_mtime'] = path.stat().st_mtime
                rows.append(row)
            except (OSError, json.JSONDecodeError):
                continue
        counts = dict(Counter(row.get('status', 'unknown') for row in rows))
        done_times = sorted(row['_mtime'] for row in rows if row.get('status') == 'complete')
        rate = None
        if len(done_times) >= 2 and done_times[-1] > done_times[0]:
            rate = (len(done_times) - 1) / ((done_times[-1] - done_times[0]) / 3600.0)
        remaining = max(0, expected - counts.get('complete', 0) - counts.get('failed', 0))
        eta_hours = remaining / rate if rate and rate > 0 else None
        eta = (datetime.fromtimestamp(time.time() + eta_hours * 3600, timezone.utc).isoformat()
               if eta_hours is not None else None)
        return {
            'expected': expected, 'records': len(rows), 'counts': counts,
            'complete': counts.get('complete', 0), 'running': counts.get('running', 0),
            'failed': counts.get('failed', 0), 'remaining': remaining,
            'rate_complete_per_hour': rate, 'eta_hours_at_observed_rate': eta_hours,
            'eta_utc_at_observed_rate': eta,
        }

    diagnostic_root = Path('/var/tmp/dora-adaptive-rank/runs/v3_deletion_diagnostics_declared_20261015_final')
    diagnostic_count = len(list(diagnostic_root.glob('*_s*_summary.json')))
    tuning = queue_status('/var/tmp/dora-adaptive-rank/runs/v3_tuning_20261015', 164)
    screening = queue_status('/var/tmp/dora-adaptive-rank/runs/v3_screening_20261015', 123)
    tuning_count = tuning['records']
    screening_count = screening['records']
    if hpo_counts.get('complete', 0) < 492:
        live_phase = 'B_hpo'
    elif tuning_count < 164:
        live_phase = 'B_tuning (diagnostics queued until tuning finishes)'
    elif diagnostic_count < 4:
        live_phase = 'B_deletion_diagnostics'
    elif screening_count < 123:
        live_phase = 'B_screening'
    else:
        live_phase = 'B_complete'
    estimate = {
        'schema': 'rankstudy-v3-roadmap', 'updated_at': now, 'first_eta_snapshot_at': anchor,
        'scope': 'All required stages retained; no reduced-scope completion claim.',
        'actual_hardware': '4 RTX PRO 6000 Blackwell Max-Q GPUs',
        'full_horizon_timing_evidence': timing,
        'pipeline_engineering_checks': pipelines,
        'live_phase': live_phase,
        'hpo_counts': hpo_counts,
        'hpo_rate_complete_per_minute': hpo_rate,
        'hpo_remaining_hours_at_observed_rate': hpo_remaining,
        'hpo_eta_utc_at_observed_rate': hpo_eta,
        'declared_diagnostics_complete': diagnostic_count,
        'tuning_queue': tuning,
        'screening_queue': screening,
        'tuning_launcher_records': tuning_count,
        'screening_launcher_records': screening_count,
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
        f"The live phase is **{live_phase}**. The corrected HPO queue currently has "
        f"**{hpo_counts.get('complete', 0)}/492 complete**, {hpo_counts.get('running', 0)} running "
        f"and {hpo_counts.get('failed', 0)} failed. No primary screening, confirmation or official "
        'benchmark comparison is complete. The 166 quarantined v2 trials remain excluded. All 41 '
        'candidates and the complete requested scope remain registered.', '',
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
        'Strict independent DDP trajectory and four BF16 refactor-parity failures remain disclosed; '
        'five FP32 model export oracles pass. See the validation report for the scope of each check.', '',
        f"Stage B contains **{len(stage_b)} mandatory training jobs**: 492 short searches, "
        '164 full-horizon tuning runs and 123 independent screening runs. Conditional rescue trials are extra. '
        f"At the two measured native rates, training alone extrapolates to **{min(gpu_hours):.1f}–{max(gpu_hours):.1f} GPU-hours**, "
        f"or **{min(ideal_hours):.1f}–{max(ideal_hours):.1f} hours** with ideal four-GPU parallelism. "
        'This arithmetic excludes evaluation, setup, pipeline overhead differences, rescue trials and remaining engineering. '
        'The timing runs used two concurrent GPUs; the live HPO rate is an observed four-job queue measurement.', '',
        (f"At the current HPO rate ({hpo_rate:.2f} completed trials/minute), the remaining short search is "
         f"estimated at {hpo_remaining:.2f} hours, near {hpo_eta}. This forecast is arithmetic from "
         'launcher completion times and is not a guarantee.') if hpo_rate is not None else
        'A live HPO rate is not yet available; the queue forecast will be populated after two completed trials.', '',
        f"The live full-horizon tuning queue has **{tuning['complete']}/{tuning['expected']} complete**, "
        f"{tuning['running']} running and {tuning['failed']} failed. "
        + (f"Its observed rate is {tuning['rate_complete_per_hour']:.1f} jobs/hour, implying "
           f"about {tuning['eta_hours_at_observed_rate']:.1f} hours remaining and an ETA near "
           f"{tuning['eta_utc_at_observed_rate']}. "
           if tuning['eta_hours_at_observed_rate'] is not None else
           'A rate-based ETA is not yet estimable. ')
        + 'The four diagnostics are held until tuning finishes because of the recorded order incident.', '',
        f"The screening queue is **{screening['complete']}/{screening['expected']} complete** "
        f"({screening['running']} running, {screening['failed']} failed). "
        + (f"Its observed-rate estimate is {screening['eta_hours_at_observed_rate']:.1f} hours."
           if screening['eta_hours_at_observed_rate'] is not None else
           'It has not started, so no measured screening-rate ETA is claimed.'), '',
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
