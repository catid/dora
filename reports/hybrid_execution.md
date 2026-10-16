# IniLoRA-alpha × AdaLoRA-SVD execution record

The exploratory hybrid was registered as `I_ADALORA_SVD` (fixed multiplier)
and `I_ADALORA_SVD_NORM` (active-rank-normalized multiplier). The primary
queue used 24 HPO trials, 8 full-horizon tuning runs and 10 fresh screening
runs (five seeds per method after the two-seed declared tuning stage). All
retry runs completed successfully.

The first HPO launch is retained at
`/var/tmp/dora-adaptive-rank/runs/v3_hybrid_hpo_20261016`; all 24 jobs failed
before model loading because seed 4201 had no sealed stream manifest. The
retry root is
`/var/tmp/dora-adaptive-rank/runs/v3_hybrid_hpo_20261016_retry`; manifests for
4201–4203 were generated from the unchanged sealed training pool and merged
into `artifacts/manifests_v2/index.json`. The initial failures are setup
failures and are excluded from quality means.

Retry roots:

- HPO: `/var/tmp/dora-adaptive-rank/runs/v3_hybrid_hpo_20261016_retry`
- Tuning: `/var/tmp/dora-adaptive-rank/runs/v3_hybrid_tuning_20261016`
- Screening seeds 3001–3003: `/var/tmp/dora-adaptive-rank/runs/v3_hybrid_screening_20261016`
- Screening seeds 3004–3005: `/var/tmp/dora-adaptive-rank/runs/v3_hybrid_screening_extra_20261016`

The rank-4 follow-up had the same missing-manifest issue for 4101–4103. Its
original 132 failed receipts are retained at
`/var/tmp/dora-adaptive-rank/runs/v3_rank4_hpo_20261016`; the repaired HPO,
tuning and screening roots use the `_retry` suffix and completed with no
training failures.
