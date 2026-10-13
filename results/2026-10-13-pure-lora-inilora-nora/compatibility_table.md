# Pure-LoRA IniLoRA-alpha + NoRA compatibility sweep

The first four rows are the locked baseline controls. The final row is selected on validation from five factor geometries and four initializer scales; values are mean ± sample SD over fresh seeds 42–44.

| Method | SciFact nDCG@10 ×100 | NFCorpus nDCG@10 ×100 | ViGGO exact match % | Aircraft macro accuracy % |
|:--|--:|--:|--:|--:|
| LoRA | 65.60 ± 0.45 | 32.16 ± 0.08 | 66.54 ± 0.81 | 60.58 ± 0.62 |
| NoRA | 68.65 ± 1.54 | 31.77 ± 0.41 | 75.65 ± 2.51 | 59.25 ± 0.73 |
| IniLoRA-alpha | 70.59 ± 1.65 | 32.34 ± 0.25 | 80.60 ± 2.29 | 59.70 ± 1.10 |
| Original IniLoRA + NoRA | 70.13 ± 1.23 | 31.88 ± 0.48 | 73.83 ± 10.58 | 57.71 ± 0.56 |
| Best compatible mix | 70.50 ± 1.75 | 32.29 ± 0.19 | 78.52 ± 5.12 | 59.34 ± 0.46 |

Selected compatibility settings:

| Task | Geometry | Scale | LR |
|:--|:--|--:|--:|
| SciFact | up_anchor | 1 | 0.001 |
| NFCorpus | up_anchor | 0.25 | 0.0001 |
| ViGGO | down_learned | 0.25 | 0.0003 |
| Aircraft | down_anchor | 0.25 | 0.001 |
