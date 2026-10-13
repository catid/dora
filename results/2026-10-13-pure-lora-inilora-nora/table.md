# Pure-LoRA IniLoRA-alpha + NoRA comparison

Validation-selected scale and learning rate; mean ± sample SD over seeds 42–44.

| Method | SciFact nDCG@10 ×100 | NFCorpus nDCG@10 ×100 | ViGGO exact match % | Aircraft macro accuracy % |
|:--|--:|--:|--:|--:|
| LoRA | 65.60 ± 0.45 | 32.16 ± 0.08 | 66.54 ± 0.81 | 60.58 ± 0.62 |
| NoRA | 68.65 ± 1.54 | 31.77 ± 0.41 | 75.65 ± 2.51 | 59.25 ± 0.73 |
| IniLoRA-alpha | 70.59 ± 1.65 | 32.34 ± 0.25 | 80.60 ± 2.29 | 59.70 ± 1.10 |
| IniLoRA-alpha + NoRA | 70.13 ± 1.23 | 31.88 ± 0.48 | 73.83 ± 10.58 | 57.71 ± 0.56 |

Selected settings:

| Task | LoRA LR | NoRA LR | IniLoRA scale / LR | IniLoRA + NoRA scale / LR |
|:--|--:|--:|--:|--:|
| SciFact | 0.001 | 0.001 | 1 / 0.001 | 0.25 / 0.001 |
| NFCorpus | 0.0003 | 0.0001 | 0.25 / 0.0001 | 1 / 0.0001 |
| ViGGO | 0.0003 | 0.0003 | 0.25 / 0.0003 | 0.5 / 0.0003 |
| Aircraft | 0.001 | 0.001 | 0.25 / 0.001 | 0.25 / 0.0003 |
