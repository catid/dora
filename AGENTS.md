# Adaptive rank study

The active specification is the user's neutral IniLoRA-alpha × adaptive rank
budgeting study (2026-10-14). This work is isolated in `src/rankstudy`, `tests`,
`configs`, `scripts`, `artifacts`, and `reports`. Preserve existing DoRA experiments.

- Treat all proposed improvements as hypotheses. Keep all 41 mandatory cells,
  failures, full-horizon screening, confirmation, and equal-compute/export branches.
- Do not publish, push, upload data, start servers, alter system configuration,
  or terminate unrelated workloads. Local commits are authorized.
- Use project `.venv`, Python 3.12, and pinned nightly CUDA 13.0 PyTorch only.
  Do not install stable torch over the nightly. Keep dependencies binary-only.
- Actual host: four RTX PRO 6000 Blackwell Max-Q GPUs, not eight B300s.
  Recheck allocation and GPU processes before each launch.
- Bulk storage: `/var/tmp/dora-adaptive-rank`; never delete existing study evidence.
- Original backbone is immutable. Centered corrections belong to the adapter.
  Use cancellation-free FP32 displacement arithmetic. Keep actual scale fixed.
- Select using dev_tune only. Seal dev_audit and official test outcomes until the
  prescribed configuration/analysis freeze. Test question text is for dedup only.
- Gate and sensitivity gradients come from the same global mean task loss,
  before clipping. Custom allocation is monotonic, deterministic, cost-aware.
- Reference AdaLoRA semantics stay distinct from custom allocator/penalty variants.
- Tests, manifests, and reports must describe what actually ran. Pending is not
  complete. Distinguish native parameter budgets from update rank and export size.
- Use `bd` if its embedded database works; otherwise `artifacts/tasks.json` is the
  fallback tracker. `artifacts/run_plan.json` and `coverage.json` cover execution.
- At continuation, read `reports/progress.md` and the task tracker first; confirm
  live process/session handles before restarting jobs.

## 2026-10-15 protocol correction

- v2 HPO is quarantined; do not restart its supervisors or select from its results.
  Evidence: `reports/protocol_v2_audit.md` and `artifacts/protocol_v2_quarantine.json`.
- Corrected implementation: `src/rankstudy/v3`, `scripts/run_one_v3.py`.
  Use project `.venv` with the existing pinned nightly build.
- Current complete-scope plan/coverage are `artifacts/run_plan_v3.json` and
  `artifacts/coverage_v3.json`; older 959-run files are superseded, not full scope.
- `artifacts/protocol_execution_gate.json` keeps large sweeps closed until all
  relevant gates are verified. Engineering smoke results never select HPO configs.
- Native EFFICIENT_ATTENTION is the recorded SDPA exception: cuDNN rejects strict
  determinism and native Flash rejects explicit padding masks. Preserve the masks.
  Actual-model efficient-SDPA profile and bitwise GPU resume receipts exist.
- Always preserve failed attempts in distinct run directories. Do not overwrite
  terminal records. Current raw reports: `reports/smoke_v3_summary.json` and
  `reports/failures_v3.csv`. Primary sweeps/confirmations remain unfinished;
  current validation scope is recorded in `reports/validation_v3.md`.

## Pipeline checkpoint and roadmap

- All 41 cells now have complete 32-update GPU smokes; the full suite passes 79
  tests. Both 512-update native timing diagnostics have finished. Verify handles
  before assuming any previously launched worker remains live.
- Common deployment export is frozen to global singular-energy allocation for
  EVERY family, including train/compress cells at their single 75% event. Native
  uniform/global cells stay distinct. Four fresh exports meet the exact factor
  payload budget; the three older uniform underspends remain preserved.
- Restart-student and compression-recovery checkpoint resumes are bitwise equal
  to uninterrupted controls. Stage attempts are immutable, process-checked and
  separately costed. Rejected invocations cannot overwrite another run's metadata.
- Full-model DDP gradients/scores/masks pass at identical checkpoints through
  wrapper/optimizer rebuilding. Independent BF16 and FP32-arithmetic trajectory
  checks FAIL strict tolerances and remain unproven. Keep primary trials on one
  GPU; do not claim cross-topology trajectory equivalence.
- Five exact-export FP32 model oracles pass. Four conservative BF16 full-model
  refactor checks fail despite small NLL shifts; preserve their measured distortion
  and do not widen thresholds after observing it or label it bitwise equivalence.
- Retained synthetic core: `/var/tmp/dora-adaptive-rank/runs/v3_synthetic_retained_20261015`,
  328 jobs / 984 budgeted runs / 1,272 stage checkpoints. Earlier unretained runs
  under `runs/v3/A_syn_*` are separately charged engineering repeats, not extra
  independent seeds. Coverage uses the retained root explicitly.
- Parser/generation, shadow deletion/curvature/gauge/restoration, complete primary
  instrumentation and the balanced HPO queue remain open. Both v3 runners enforce
  `protocol_execution_gate.json` for non-smoke calls. Do not bypass it.
