# Audited initialization-study export

This directory is the repository-facing export of the completed initialization study.

- `main_report.md` contains all centered, warm-start, and quantized family/lane tables.
- `confirmation_report.md` contains the locked seeds 45–49 confirmation roles.
- `all_families.csv`, `all_comparisons.csv`, and `search_budgets.csv` retain the complete centered grid.
- `overview.png` and `overview.svg` show accepted DoRA, the Gaussian control, and each validation-selected winner for both seed groups.
- `main_replay_audit.json`, `confirmation_replay_audit.json`, and `extension_replay_audit.json` record portable saved-prediction replay.
- `SHA256SUMS` covers every file in this directory except itself.

Primary results use seeds 42–44. Fresh confirmation uses only seeds 45–49 and is not pooled with the primary results. Means use sample standard deviations; intervals are exploratory paired bootstrap intervals without multiplicity correction. Portable replay omits tensor payloads while retaining their original hashes in the evidence manifests.
