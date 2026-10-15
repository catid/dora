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
- `src/rankstudy/v3` has 30 new protocol tests plus four run-plan tests (65 total tests pass). Tests
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
- Thirteen smoke executions have completed; four engineering failures are
  retained. No smoke result is eligible for primary configuration selection.
- Two 512-update full-horizon diagnostics are active at common LR=1e-4/scale=1
  under `/var/tmp/dora-adaptive-rank/runs/v3_full_horizon_checks_20261015`:
  L_FIXED PID 3346030 on GPU0 and I_GATE PID 3346031 on GPU2. Verify `/proc` before
  relying on these handles. They use only 16 tune examples for diagnostic loss;
  they are not HPO or independent screening trials.
- v3 large sweeps remain disabled by `artifacts/protocol_execution_gate.json`.
  No full-horizon method comparison, independent screening/confirmation, official
  benchmark evaluation, export/equal-compute result, 7B result or winner exists.

Next: finish full-horizon diagnostics, complete remaining DDP and Stage A checks,
corrected restart/compression runners and tests; only then restart the balanced
41-method HPO. Preserve all unsuccessful attempts and account for their cost.
