| Method | Flowers102 train+val (s) | SciFact train (s) | ViGGO train+val (s) | SDXL train (s) |
|:--|--:|--:|--:|--:|
| Task baseline | 29.2 | — | — | — |
| LoRA | 55.0 | 1.6 | 59.2 | 135.8 |
| DoRA | 68.1 | 1.8 | 61.0 | 184.6 |
| NoRA | 55.6 | 1.7 | 59.7 | 180.6 |
| DoRA+NoRA | 68.9 | 1.9 | 61.5 | 229.4 |

Median recorded final-training wall time across three seeds; excludes downloads, model/setup work, and generated-image/text evaluation. Vision and JSON extraction include their scheduled validation passes and checkpoint bookkeeping. Retrieval and SDXL record training loops only. Learning-rate search trials are additional work and are retained separately in the raw artifacts. All task budgets are fixed within a task; these seconds do not measure time to equal quality. Frozen baselines have no training; the vision baseline trains its classifier.
