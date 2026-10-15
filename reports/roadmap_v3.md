# Roadmap and provisional ETA

Updated 2026-10-15T05:39:35.166627+00:00.

The study remains in correctness validation. No corrected primary HPO, screening, confirmation or official benchmark comparison is complete. The 166 quarantined v2 trials remain excluded. All 41 candidates and the complete requested scope remain registered.

| Completed full-horizon engineering run | Updates | Input tokens | Charged training minutes |
|---|---:|---:|---:|
| I_GATE | 512 | 4,275,819 | 6.85 |
| L_FIXED | 512 | 4,275,819 | 6.24 |

These runs used a common untuned configuration and only 16 development groups. They establish execution and timing, not a method ranking.

| Milestone | Provisional cumulative ETA from this snapshot |
|---|---|
| Remaining Stage A checks and corrected queue | 24-48 hours |
| Stage B: all 41 candidates with independent screening seeds | 2-4 days |
| Stages C/D: fresh-seed, rank-four, export and equal-compute comparisons | 4-7 days |
| Stage E plus official evaluations and final audit/report | 1-3 weeks |

Stage A remaining work includes complete GPU pipeline coverage and interruption recovery, separate-process export parity/distortion checks, full-model DDP and compaction rebuild, parser/evaluation validation, synthetic controls and diagnostic state restoration.

Stage B contains **779 mandatory training jobs**: 492 short searches, 164 full-horizon tuning runs and 123 independent screening runs. Conditional rescue trials are extra. At the two measured native rates, training alone extrapolates to **42.6–46.8 GPU-hours**, or **10.7–11.7 hours** with ideal four-GPU parallelism. This arithmetic excludes evaluation, setup, pipeline overhead differences, rescue trials and remaining engineering. The timing runs used two concurrent GPUs; four-job queue throughput still needs calibration.

Planning estimates, not guarantees. Assume continued four-GPU access. 7B and generation throughput are not yet calibrated; correctness failures can extend these ranges.

The 3,064-entry plan includes diagnostics, conditional branches and separate evaluation jobs; it is not a count of unconditional training runs. See `artifacts/coverage_v3.json` for exact coverage and `reports/progress.md` for validation evidence.

Regenerate: `.venv/bin/python scripts/report_roadmap_v3.py`.
