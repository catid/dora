# Adaptive-rank study progress

Updated 2026-10-15 UTC. The full study remains active and incomplete.

The actual host has four RTX PRO 6000 Blackwell Max-Q GPUs. Project `.venv` uses
Python 3.12 and pinned PyTorch `2.16.0.dev20261013+cu130`. Corrected data v2 retains
20,000 intact training examples, 512 tune groups and 512 sealed audit groups.
Official benchmark outcomes and dev_audit remain sealed.

Completed evidence:

- **79 automated tests pass** (`artifacts/tests_v3_stageA_final_20261015.xml`),
  including true displacement algebra, score reductions, exact budgets, original
  AdaLoRA trajectory parity, compaction, safe checkpointing and pipeline controls.
- **All 41 registry cells completed 32-update Qwen1.5B GPU smokes**, with identical
  token exposure and unchanged original backbone. Wide controls and faithful
  AdaLoRA's E-scalar excess are explicitly distinguished from budget-matched cells.
  Evidence: `artifacts/v3_registry_smoke_audit.json`. These are engineering trials.
- Two 512-update diagnostics completed: L_FIXED charged 374.140 GPU-seconds and
  I_GATE 410.880 GPU-seconds. Each consumed 4,275,819 input tokens and met the final
  scalar budget. Only 16 tune groups were evaluated; neither is a primary result.
- **Common deployment export now uses global singular energy for every family.**
  Four fresh real-model exports each retain 1,376,256 ordinary-factor parameters /
  5,505,024 factor bytes. File-header/metadata bytes are reported separately.
  The three initial uniform-export underspends remain preserved and excluded from
  equal-size claims (`artifacts/v3_common_export_budget_audit.json`).
- Restart-student and compression-recovery interruptions reproduce final factors,
  Adam state, score state, RNG and exposure bit for bit. Attempts are stored
  separately; completed history is not double-charged, and repeated/failed work
  remains visible. Evidence: `artifacts/v3_pipeline_resume_audit.json`.
- **328 retained synthetic jobs / 984 budgeted runs** completed across all 41
  methods, two data seeds, isotropic/anisotropic inputs, noisy/clean targets and
  three feasible budgets. All 1,272 stage checkpoints passed payload verification.
  Paired dataset hashes agree. Known-teacher oracle, seed-level results, rank maps,
  spectra and cost accounting are in `reports/synthetic_v3.md` and its CSVs.
  Earlier engineering repeats are charged separately, not additional replications.

Numerical findings and limits:

- Native efficient SDPA is the recorded backend exception: cuDNN rejects strict
  determinism, and native Flash rejects the required explicit padding masks.
  Forward/gradient attention oracles, profiling and bitwise native GPU resume pass.
- All five representation families pass separate-process exact-refactor checks
  with FP32 arithmetic on the same stored BF16 backbone values. This is an oracle,
  not a change to primary BF16 training. Four conservative BF16 refactor checks
  fail their strict full-model logit tolerance; their small probe NLL changes and
  truncated-compression damage remain explicitly reported. No thresholds were widened.
- Tiny-layer DDP/NCCL parity passes ten cases. Real-model DDP passes sixteen
  same-checkpoint checks through unequal local token counts, no_sync accumulation,
  mask broadcast, compaction and wrapper/optimizer rebuilding. Independently
  evolving BF16 and FP32-arithmetic trajectory checks fail strict tolerances.
  Cross-topology trajectory equivalence is unproven; primary trials stay on one GPU.
- `reports/validation_v3.md` and `reports/export_validation_v3.csv` retain the
  positive and negative checks. They do not establish a method-quality winner.

Current declared coverage is **331 passed / 2,733 pending** out of 3,064 entries
(including conditional/evaluation jobs). Most engineering diagnostics are accounted
separately. All primary HPO, independent screening, confirmation, export/equal-compute
quality comparisons, 7B and heterogeneous-module studies remain pending.

Next: finish parser/generation validation, shadow deletion/curvature/gauge and
state-restoration diagnostics, primary metric instrumentation and the balanced
v3 HPO queue. Both runners enforce the closed primary-execution gate; smoke labels
must never be used to select configurations or bypass the required validation.

The v1 question/response mismatch and all 166 quarantined v2 HPO trials remain
preserved and excluded. No v2 supervisors should be restarted. Preserve unrelated
DoRA/NoRA files; local commits are allowed, publishing/pushing is not.

The [roadmap](roadmap_v3.md) retains its original ETA anchor: provisional 2–4 days
to corrected Stage B screening and 1–3 weeks to the complete study with continued
four-GPU access. Generation and 7B throughput remain uncalibrated.
