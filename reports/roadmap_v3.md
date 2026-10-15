# Roadmap and provisional ETA

Updated 2026-10-15T11:18:54.327077+00:00.

Planning ranges remain anchored to the initial ETA snapshot, 2026-10-15T05:39:35.166627+00:00; refreshing this report does not restart the clock.

The live phase is **B_hpo**. The corrected HPO queue currently has **205/492 complete**, 4 running and 0 failed. No primary screening, confirmation or official benchmark comparison is complete. The 166 quarantined v2 trials remain excluded. All 41 candidates and the complete requested scope remain registered.

| Completed full-horizon engineering run | Updates | Input tokens | Charged training minutes |
|---|---:|---:|---:|
| I_GATE | 512 | 4,275,819 | 6.85 |
| L_FIXED | 512 | 4,275,819 | 6.24 |
| L_GATE | 512 | 4,275,819 | 6.24 |

These runs used a common untuned configuration and only 16 development groups. They establish execution and timing, not a method ranking.

| Milestone | Provisional cumulative ETA from initial snapshot |
|---|---|
| Remaining Stage A checks and corrected queue | 24-48 hours |
| Stage B: all 41 candidates with independent screening seeds | 2-4 days |
| Stages C/D: fresh-seed, rank-four, export and equal-compute comparisons | 4-7 days |
| Stage E plus official evaluations and final audit/report | 1-3 weeks |

Stage A has now executed all 41 GPU smoke cells and 984 retained synthetic budgeted runs. Pipeline resume, common export allocation and same-checkpoint DDP/rebuild checks have receipts. Strict independent DDP trajectory and four BF16 refactor-parity failures remain disclosed; five FP32 model export oracles pass. See the validation report for the scope of each check.

Stage B contains **779 mandatory training jobs**: 492 short searches, 164 full-horizon tuning runs and 123 independent screening runs. Conditional rescue trials are extra. At the two measured native rates, training alone extrapolates to **42.6–46.8 GPU-hours**, or **10.7–11.7 hours** with ideal four-GPU parallelism. This arithmetic excludes evaluation, setup, pipeline overhead differences, rescue trials and remaining engineering. The timing runs used two concurrent GPUs; the live HPO rate is an observed four-job queue measurement.

At the current HPO rate (1.92 completed trials/minute), the remaining short search is estimated at 2.49 hours, near 2026-10-15T13:48:32.790402+00:00. This forecast is arithmetic from launcher completion times and is not a guarantee.

Planning estimates, not guarantees. Assume continued four-GPU access. 7B and generation throughput are not yet calibrated; correctness failures can extend these ranges.

The 3,064-entry plan includes diagnostics, conditional branches and separate evaluation jobs; it is not a count of unconditional training runs. See `artifacts/coverage_v3.json` for exact coverage and `reports/progress.md` for validation evidence.

Regenerate: `.venv/bin/python scripts/report_roadmap_v3.py`.
