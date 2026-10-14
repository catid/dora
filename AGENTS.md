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
