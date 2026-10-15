# Adaptive-rank study progress

Updated 2026-10-15 UTC.

The full user protocol remains active; the study is incomplete.

- Actual host: four RTX PRO 6000 Blackwell Max-Q GPUs. Pinned Python 3.12 /
  PyTorch `2.16.0.dev20261013+cu130` are available in project `.venv`.
- Corrected data v2 retains 20,000 intact training rows, 512 tune groups and 512
  sealed audit groups, with exact paired effective-update manifests.
- v1 results remain withdrawn for question/response mismatches.
- All 166 terminal v2 HPO records are now quarantined from primary selection.
  See `protocol_v2_audit.md`; the v2 supervisors and four verified study workers
  were stopped without deleting evidence. The older 959-job scope was incomplete.
- The full automated suite now passes 71 tests, including six new pipeline tests
  (`artifacts/tests_v3_20261015_pipeline.xml`). Tests
  include original AdaLoRA optimizer-trajectory parity, true displacements,
  gradients, score accumulation, cost allocation, no momentum drift, compaction,
  and safe CPU resume. These do not certify the entire study.
- Eight real Qwen1.5B GPU executions (including one resume) finished under the
  initial cuDNN smoke policy. All initial representations trained and kept the
  original backbone hash. Custom allocation physically compacted within budget.
  Reference AdaLoRA's extra E scalars are explicitly reported.
- The cuDNN resume diagnostic did NOT produce identical final parameters/masks.
  First resumed forward loss matched; backward gradients differed. Do not call
  this exact resume. The receipt is `artifacts/v3_gpu_resume_audit.json`.
- A deterministic SDPA probe finds cuDNN rejects strict deterministic execution;
  native Flash SDPA rejects an explicit padding mask. Native EFFICIENT_ATTENTION
  passes short FP64-oracle forward/gradient checks and bitwise repeat checks.
  This exception is recorded in `artifacts/sdpa_v3_oracle.json`. A Qwen profile
  verified native efficient SDPA, and the 32-update interrupted/resumed run now
  matches adapter tensors, optimizer state, score EMA, RNGs and pruning events
  bit for bit, including physical compaction. Receipt:
  `artifacts/v3_deterministic_resume_audit.json`.
- Strict deterministic CE initially rejected the old 3D NLLLoss2d path. Four
  failed smoke attempts are retained. The mathematically equivalent flattened
  token CE passes output/gradient tests and strict deterministic CUDA execution.
- Two-rank DDP/NCCL tiny-layer parity passed ten cases across L/I scores and T0/T1
  gates, including unequal token counts, signed gate reduction, ACT statistics,
  and masks. Full-model DDP and DDP compaction/rebuild remain separate open checks.
- Twenty-three engineering executions have completed (15 native, eight pipeline);
  four failed attempts remain retained. No smoke result is eligible for primary
  configuration selection. Regenerated records are in `reports/smoke_v3.csv`.
- Both 512-update full-horizon diagnostics completed at common LR=1e-4/scale=1
  under `/var/tmp/dora-adaptive-rank/runs/v3_full_horizon_checks_20261015`.
  L_FIXED charged 374.140 GPU-seconds; I_GATE charged 410.880 GPU-seconds. Both
  consumed 4,275,819 input tokens, met the exact final scalar budget and preserved
  the original backbone hash. They use only 16 tune groups for diagnostic loss;
  they are not HPO or independent screening trials. Their old PIDs have exited.
- Eight 32-update pipeline executions completed under
  `/var/tmp/dora-adaptive-rank/runs/v3_pipeline_smoke_20261015`: cross-initializer
  restarts in both directions, L/uniform and I/global train-compress-recover,
  plus common export for a restart, T0, T1+penalty and reference AdaLoRA.
  All consumed exactly 267,916 input / 199,831 supervised tokens without stage
  overlap, preserved the original backbone, and remained within the upper budget.
  Ordinary exports were loaded in separate recovery child processes. This does
  not yet prove immediate full-model export parity or interrupted pipeline resume.
- **Open export fairness issue:** uniform compression currently caps a module's
  exported rank at its available thin-factor spectrum length. Three common-export
  smokes therefore underspend the deployment budget: restart 362,496 scalars,
  T0 36,864 and T1+penalty 258,048. Preserve these runs, fix/predeclare common
  export allocation before primary execution, and test actual representation size.
  Do not present these as equal-export-size results. Evidence:
  `artifacts/v3_pipeline_smoke_audit.json`.
- v3 large sweeps remain disabled by `artifacts/protocol_execution_gate.json`.
  No full-horizon method comparison, independent screening/confirmation, official
  benchmark evaluation, export/equal-compute result, 7B result or winner exists.

Next: resolve common-export budget equality and pipeline interruption recovery;
complete full-model DDP/rebuild, parser/evaluation, synthetic and diagnostic
restoration checks; then enable the balanced v3 41-method queue. Current plan
coverage is two completed declared diagnostics out of 3,064 entries, with all
other entries pending; narrow tests and engineering smokes are accounted separately.

The [roadmap](roadmap_v3.md) gives a provisional 2–4 days to corrected Stage B
screening and 1–3 weeks for the complete study, assuming continued four-GPU
access. Training-only arithmetic is 10.7–11.7 ideal four-GPU hours for Stage B;
evaluation, engineering and later stages are additional. 7B and generation
throughput are still uncalibrated, so the full-study estimate remains uncertain.
