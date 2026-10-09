| Method | FGVC-Aircraft | NFCorpus retrieval | SciFact transfer | COGS generalization |
|:--|:--|:--|:--|:--|
| Task baseline | 43.57 ± 1.01 | 31.68 | 64.63 | 8.18 |
| LoRA | 64.38 ± 0.19 | 32.20 ± 0.20 | 62.64 ± 0.60 | 67.01 ± 3.06 |
| DoRA | 64.55 ± 0.81 | 32.24 ± 0.15 | 62.66 ± 0.59 | 66.47 ± 3.72 |
| NoRA | 65.03 ± 0.53 | 32.17 ± 0.30 | 62.11 ± 0.52 | 70.59 ± 4.59 |
| DoRA+NoRA | 64.58 ± 0.95 | 32.17 ± 0.38 | 62.12 ± 0.34 | 66.82 ± 4.39 |
| DoRA+NoRA (slow magnitudes) | 63.60 ± 1.47 | 32.20 ± 0.40 | 62.14 ± 0.49 | 67.56 ± 4.84 |
| DoRA+NoRA (input gains) | 64.64 ± 0.94 | 31.31 ± 0.46 | 58.28 ± 0.51 | 67.36 ± 2.68 |

Values are mean ± sample standard deviation across three training seeds; frozen baselines have one deterministic evaluation. Aircraft uses three trained-head baseline seeds. All metrics are on a 0–100 scale; higher is better. These are separate tasks, so no cross-task average or winner is computed.

Aircraft: macro accuracy. NFCorpus/SciFact: nDCG@10 ×100. COGS: atom-set exact match on the balanced 672-example OOD sample (32 per category).
