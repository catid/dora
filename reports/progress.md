# Adaptive-rank study progress

Updated 2026-10-14 UTC.

- Environment: pinned PyTorch nightly `2.16.0.dev20261013+cu130`, CUDA 13.0, four healthy RTX PRO 6000 Blackwell Max-Q GPUs.
- References audited and source revisions recorded.
- Stage A algebra/allocator/data/frozen-state tests: 20 passed.
- Data preparation: complete deduplicated pool has 13,928 connected groups; after 512+512 development groups, 12,904 train groups remain. Requested 20,000 train groups is blocked by the pinned MetaMathQA revision; no duplicates were introduced.
- Real GPU smoke: standard L, centered I, T0 triplet, online RAW, and faithful AdaLoRA completed.
- Faithful AdaLoRA now uses A/B Normal(0, 0.02), zero E, fixed `ranknum + 1e-5`, element-score/uncertainty allocation, and separate reference metadata.
- Superseded native HPO: 348/348 short trials completed, retained under `/var/tmp/dora-adaptive-rank/runs/hpo_native2`, but excluded from selection after the token-exposure audit. See `reports/protocol_invalid.md`.
- Accepted native HPO with global token normalization, faithful AdaLoRA post-step ordering, monotonic custom masks, and microbatch-1 memory policy is running under `/var/tmp/dora-adaptive-rank/runs/hpo_accepted_final2`. Selection remains pending until all 348 rows finish. ACT rows use a deterministic maximum of 256 valid tokens per effective update/module.
- A resumable continuation (`scripts/continue_native_study.sh`) is waiting on HPO completion and then launches full-horizon tuning (two configs × seeds 1002/1003) and three-seed screening (2001–2003) for all executable native cells.
- Restart, compression/recovery, export, official GSM8K/MATH, 7B, heterogeneous-module, DDP, and report-generation stages remain pending or blocked until the corrected native selection and their implementations are complete.
