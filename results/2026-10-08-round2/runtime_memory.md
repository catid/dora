| Task | Rank | Method | Trainable parameters | Training seconds, mean ± SD | Peak allocated GiB, max |
|:--|--:|:--|--:|--:|--:|
| aircraft | 2 | Task baseline | 76,900 | 139.68 ± 1.75 | 1.034 |
| aircraft | 2 | LoRA | 371,812 | 169.63 ± 0.65 | 8.958 |
| aircraft | 2 | DoRA | 454,756 | 192.46 ± 0.38 | 13.043 |
| aircraft | 2 | NoRA | 371,812 | 170.36 ± 0.46 | 8.958 |
| aircraft | 2 | DoRA+NoRA | 454,756 | 194.96 ± 0.90 | 13.043 |
| aircraft | 2 | DoRA+NoRA (slow magnitudes) | 454,756 | 193.56 ± 1.38 | 13.043 |
| aircraft | 2 | DoRA+NoRA (input gains) | 519,268 | 193.68 ± 0.73 | 13.045 |
| aircraft | 8 | Task baseline | 76,900 | 150.13 ± 3.55 | 1.034 |
| aircraft | 8 | LoRA | 1,256,548 | 166.70 ± 0.42 | 8.983 |
| aircraft | 8 | DoRA | 1,339,492 | 189.22 ± 2.08 | 13.068 |
| aircraft | 8 | NoRA | 1,256,548 | 167.13 ± 2.41 | 8.983 |
| aircraft | 8 | DoRA+NoRA | 1,339,492 | 188.27 ± 4.08 | 13.068 |
| aircraft | 8 | DoRA+NoRA (slow magnitudes) | 1,339,492 | 187.84 ± 3.18 | 13.068 |
| aircraft | 8 | DoRA+NoRA (input gains) | 1,404,004 | 188.88 ± 4.96 | 13.071 |
| nfcorpus | 2 | Task baseline | 0 | — | — |
| nfcorpus | 2 | LoRA | 18,432 | 18.77 ± 0.12 | 3.226 |
| nfcorpus | 2 | DoRA | 23,040 | 22.05 ± 0.14 | 3.951 |
| nfcorpus | 2 | NoRA | 18,432 | 19.14 ± 0.07 | 3.227 |
| nfcorpus | 2 | DoRA+NoRA | 23,040 | 21.75 ± 0.97 | 3.951 |
| nfcorpus | 2 | DoRA+NoRA (slow magnitudes) | 23,040 | 22.58 ± 0.01 | 3.951 |
| nfcorpus | 2 | DoRA+NoRA (input gains) | 27,648 | 22.87 ± 0.11 | 3.952 |
| nfcorpus | 8 | Task baseline | 0 | — | — |
| nfcorpus | 8 | LoRA | 73,728 | 20.11 ± 0.58 | 3.238 |
| nfcorpus | 8 | DoRA | 78,336 | 22.32 ± 0.05 | 3.963 |
| nfcorpus | 8 | NoRA | 73,728 | 20.91 ± 0.06 | 3.238 |
| nfcorpus | 8 | DoRA+NoRA | 78,336 | 22.61 ± 0.04 | 3.964 |
| nfcorpus | 8 | DoRA+NoRA (slow magnitudes) | 78,336 | 22.48 ± 0.17 | 3.964 |
| nfcorpus | 8 | DoRA+NoRA (input gains) | 82,944 | 22.78 ± 0.06 | 3.964 |
| cogs | 8 | Task baseline | 0 | — | — |
| cogs | 8 | LoRA | 1,843,200 | 208.79 ± 1.12 | 44.175 |
| cogs | 8 | DoRA | 1,926,144 | 214.85 ± 1.08 | 45.330 |
| cogs | 8 | NoRA | 1,843,200 | 206.22 ± 1.29 | 44.176 |
| cogs | 8 | DoRA+NoRA | 1,926,144 | 216.61 ± 0.36 | 45.331 |
| cogs | 8 | DoRA+NoRA (slow magnitudes) | 1,926,144 | 213.00 ± 0.43 | 45.331 |
| cogs | 8 | DoRA+NoRA (input gains) | 2,073,600 | 213.42 ± 0.52 | 45.337 |

Timing scopes differ by task. Aircraft includes per-epoch validation and disk saves of improved checkpoints. COGS includes periodic validation and copies of the best state; its timer excludes final checkpoint save/restore and generation. Retrieval training time ends before the adapter save and excludes validation/test retrieval. Model loading and test inference are excluded throughout; compare recorded training times within a task. Retrieval has one training job for both retrieval datasets. Peak allocated memory differs from device-reserved memory. Parameter counts include the Aircraft classifier and extra input gains where applicable.

Retrieval inference was measured but the first native frozen baseline incurred shape-specific startup cost; those cold/warm timings are retained in raw records and are not used for an inference-efficiency claim.
