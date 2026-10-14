# Adaptive-rank study progress

Updated 2026-10-14 UTC.

- Environment: pinned PyTorch nightly `2.16.0.dev20261013+cu130`, CUDA 13.0, four healthy RTX PRO 6000 Blackwell Max-Q GPUs.
- References audited and source revisions recorded.
- Stage A algebra/allocator/data/frozen-state tests: 20 passed.
- Corrected v2 data preparation: 20,000 intact training rows, 512 `dev_tune` groups, and 512 disjoint `dev_audit` groups. The training rows span 8,899 connected groups; 10 groups are excluded by the held-out near-overlap filter. The requested row count is met; group count is reported separately.
- Real GPU smoke: standard L, centered I, T0 triplet, online RAW, and faithful AdaLoRA completed.
- Faithful AdaLoRA now uses A/B Normal(0, 0.02), zero E, fixed `ranknum + 1e-5`, element-score/uncertainty allocation, and separate reference metadata.
- All earlier native screens under `/var/tmp/dora-adaptive-rank/runs/hpo_native2`, `/var/tmp/dora-adaptive-rank/runs/hpo_corrected`, `/var/tmp/dora-adaptive-rank/runs/hpo_final`, and `/var/tmp/dora-adaptive-rank/runs/hpo_accepted_final2` are withdrawn from selection because they use the superseded v1 data seal; they remain preserved with the audit in `reports/protocol_invalid.md`.
- Corrected v2 native HPO is live under `/var/tmp/dora-adaptive-rank/runs/hpo_v2_seed1001`: 24/348 trials are complete and four workers are currently active. The first complete fixed cells are `L_FIXED` best dev NLL 0.213907 (LR 1e-3, scale 4) and `I_FIXED` 0.216248 (LR 1e-4, scale 4); these are HPO screening values only.
- The HPO crossed six bookkeeping-only source commits. [`artifacts/v2_provenance_audit.json`](../artifacts/v2_provenance_audit.json) hashes the native training executable and adapter/data/allocator files at every recorded revision; all common training-file hashes are identical.
- `scripts/continue_adaptive_v2.sh` is a resumable supervisor that waits for corrected HPO, then launches corrected full-horizon tuning, fresh screening, pipeline HPO, and pipeline tuning. No later stage is counted until its immutable run records exist.
- Restart, compression/recovery, export, official GSM8K/MATH, 7B, heterogeneous-module, DDP, and report-generation stages remain pending or blocked until the corrected native selection and their implementations are complete.
