# Adaptive-rank study progress

Updated 2026-10-14 UTC.

- Environment: pinned PyTorch nightly `2.16.0.dev20261013+cu130`, CUDA 13.0, four healthy RTX PRO 6000 Blackwell Max-Q GPUs.
- References audited and source revisions recorded.
- Stage A algebra/allocator/data/frozen-state tests: 20 passed.
- Corrected v2 data preparation: 20,000 intact training rows, 512 `dev_tune` groups, and 512 disjoint `dev_audit` groups. The training rows span 8,899 connected groups; 10 groups are excluded by the held-out near-overlap filter. The requested row count is met; group count is reported separately.
- Real GPU smoke: standard L, centered I, T0 triplet, online RAW, and faithful AdaLoRA completed.
- Faithful AdaLoRA now uses A/B Normal(0, 0.02), zero E, fixed `ranknum + 1e-5`, element-score/uncertainty allocation, and separate reference metadata.
- All earlier native screens under `/var/tmp/dora-adaptive-rank/runs/hpo_native2`, `/var/tmp/dora-adaptive-rank/runs/hpo_corrected`, `/var/tmp/dora-adaptive-rank/runs/hpo_final`, and `/var/tmp/dora-adaptive-rank/runs/hpo_accepted_final2` are withdrawn from selection because they use the superseded v1 data seal; they remain preserved with the audit in `reports/protocol_invalid.md`.
- Corrected v2 native HPO is live under `/var/tmp/dora-adaptive-rank/runs/hpo_v2_seed1001`: 48/348 trials have terminal metadata and four corrected AdaLoRA workers are currently active. The best completed values are `L_FIXED` dev NLL 0.213907 (LR 1e-3, scale 4), `I_FIXED` 0.216248 (LR 1e-4, scale 4), `I_WIDE` 0.215339 (LR 1e-4, scale 4), and `L_WIDE` 0.210417 (LR 1e-3, scale 4); these are one-seed HPO screening values only. The live machine-readable matrix is [`reports/hpo_v2_live.csv`](hpo_v2_live.csv), and the refreshed coverage checkpoint records 48 complete / 911 pending runs.
- The frozen-matrix Stage A control completed 16 CPU runs across two seeds, both initializations, isotropic/anisotropic inputs, noisy/clean targets, and slot budgets 8/16/32. Its immutable output is [`artifacts/synthetic_controls_v2.json`](../artifacts/synthetic_controls_v2.json); these mechanistic errors are not LLM quality measurements.
- Before AdaLoRA produced a terminal record, an audit found that the first launcher would permanently freeze masked `E` slots, unlike upstream's post-optimizer update/reactivation behavior. Those in-flight trials were stopped and left as nonterminal evidence; the corrected reference path is now running from commit `d433047`, with the 47 completed baseline records retained.
- The HPO crossed six bookkeeping-only source commits. [`artifacts/v2_provenance_audit.json`](../artifacts/v2_provenance_audit.json) hashes the native training executable and adapter/data/allocator files at every recorded revision; all common training-file hashes are identical.
- `scripts/continue_adaptive_v2.sh` is a resumable supervisor that waits for corrected HPO, then launches corrected full-horizon tuning, fresh screening, pipeline HPO, and pipeline tuning. No later stage is counted until its immutable run records exist.
- Restart, compression/recovery, export, official GSM8K/MATH, 7B, heterogeneous-module, DDP, and report-generation stages remain pending or blocked until the corrected native selection and their implementations are complete.
