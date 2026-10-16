# Adaptive-rank study progress

Updated 2026-10-16 UTC. The full study remains active and incomplete.

## Live refresh

The earlier Stage-B status below is superseded by the completed receipts now on
disk: all 492 HPO trials, all 164 full-horizon tuning runs, all 123 screening
runs, and all four declared deletion diagnostics are complete with zero launcher
failures. Stage C's 55 representative confirmation runs are also complete with
five seeds each and zero failures; the receipt audit is
`artifacts/confirmation_audit_v3.json`. The confirmation report is
`reports/confirmation_summary.csv`. Matched Stage-C controls have just started
under `/var/tmp/dora-adaptive-rank/runs/v3_controls_20261016`; their 90 declared
runs are still pending/running and are not included in the representative
summary. Export/equal-compute, rank-4, 7B, heterogeneous-module, and official
evaluation stages remain pending.

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
- **Stage A gate passed.** Exact token-progress deletion diagnostics cover both
  `L_GATE` and `I_GATE` at 20/50/75%, with six restoration-verified receipts.
  Signed toy deletion, curvature, interaction, and gauge probes are recorded
  separately. Evidence: `artifacts/stageA_gate_v3.json`,
  `artifacts/deletion_diagnostics_v3_receipt.json`, and
  `artifacts/mechanism_diagnostics_v3.json`.
- The greedy generation/parser runner completed a public GSM8K TRAIN preflight
  with zero generation errors and zero parser timeouts. The 64-token smoke is
  intentionally not a quality result; official and audit answers remain sealed.
  Evidence: `artifacts/generation_preflight_v3.json`.
- Native receipts now include clipping counts, effective update norms, parameter
  norms, score overhead, and peak memory. Fixed, ACT, triplet-regularized, and
  faithful AdaLoRA 32-update smokes completed successfully.
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

The corrected balanced 492-trial v3 short HPO is now **492/492 complete** with no
failures. Full-horizon tuning is active: launcher receipts currently show
**57/164 complete, four running, and zero failed**. The guarded continuation
watcher holds the four required deletion diagnostics until tuning has released
the GPUs, then launches the 123 fresh screening trials. No tuning or screening
value is being presented as a final method result before the diagnostic gate and
selection freeze. Confirmation, export/compute, 7B, heterogeneous-module, and
official evaluation stages follow. Smoke labels remain ineligible for selection.

The v1 question/response mismatch and all 166 quarantined v2 HPO trials remain
preserved and excluded. No v2 supervisors should be restarted. Preserve unrelated
DoRA/NoRA files; local commits are allowed, publishing/pushing is not.

The first eight v3 HPO trials completed at 97--127 GPU-seconds each with all four
GPUs occupied. The 492-trial short HPO is currently estimated at 3--5 wall hours;
Stage B through independent screening at 12--18 wall hours after HPO, excluding
rescues and failure tails. The complete study remains approximately 1--3 weeks
with continued four-GPU access; 7B and official generation throughput remain
uncalibrated.

The registered second diagnostic seed pair (`2901`, `2902`) is still pending. A
first concurrent launch put two diagnostics on each of two GPUs and failed with
CUDA OOM before scoring; those four attempts are preserved in
`artifacts/deletion_diagnostics_declared_oom_20261015.json` and will be rerun
sequentially after tuning releases a device. The live queue and ETA are recorded
in `artifacts/roadmap_v3.json` and `reports/roadmap_v3.md`.
# Live update 2026-10-16 — hybrid and rank-4 follow-ups

The 90 matched Stage-C control jobs are complete with zero failures. The
repaired rank-4 follow-up completed 132/132 HPO, 44/44 tuning and 33/33
screening jobs; its original 132 missing-manifest failures are retained under
the non-retry root. The exploratory IniLoRA-alpha × AdaLoRA-SVD hybrid also
completed 24/24 HPO, 8/8 tuning and 10/10 screening jobs after the same
manifest repair. Five-seed endpoint means are 0.201743 ± 0.000168 for fixed
scale and 0.203984 ± 0.000409 for active-rank normalization. The fixed hybrid
is below `I_FIXED` and `I_UNIFORM_SVD`, slightly above `T1_GATE`, and well
above `L_GATE`; it is not a practical overall winner in this screen.

Raw hybrid coverage and setup failures are in `artifacts/coverage_hybrid.json`
and `reports/hybrid_execution.md`. The older status paragraphs below retain
their historical snapshots and should not be read as current queue counts.
