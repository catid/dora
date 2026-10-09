| Method | FGVC-Aircraft | NFCorpus retrieval | SciFact transfer |
|:--|:--|:--|:--|
| Task baseline | 43.57 ± 1.01 | 31.68 | 64.63 |
| LoRA | 60.35 ± 0.34 | 31.98 ± 0.23 | 61.12 ± 0.11 |
| DoRA | 61.07 ± 0.19 | 32.01 ± 0.21 | 61.38 ± 0.25 |
| NoRA | 60.34 ± 0.39 | 31.90 ± 0.65 | 59.99 ± 0.83 |
| DoRA+NoRA | 60.24 ± 0.51 | 31.92 ± 0.55 | 60.12 ± 0.96 |
| DoRA+NoRA (slow magnitudes) | 57.95 ± 1.10 | 31.89 ± 0.57 | 60.14 ± 0.89 |
| DoRA+NoRA (input gains) | 59.99 ± 0.43 | 31.87 ± 0.50 | 60.02 ± 0.94 |

Values are mean ± sample standard deviation across three training seeds; frozen baselines have one deterministic evaluation. Aircraft uses three trained-head baseline seeds. All metrics are on a 0–100 scale; higher is better. These are separate tasks, so no cross-task average or winner is computed.

Aircraft: macro accuracy. NFCorpus/SciFact: nDCG@10 ×100. COGS: atom-set exact match on the balanced 672-example OOD sample (32 per category).
