| Method | Flowers102 (GiB) | SciFact (GiB) | ViGGO (GiB) | SDXL (GiB) |
|:--|--:|--:|--:|--:|
| Task baseline | 0.71 | — | — | — |
| LoRA | 4.76 | 0.90 | 20.48 | 7.60 |
| DoRA | 6.80 | 1.09 | 20.93 | 7.60 |
| NoRA | 4.76 | 0.90 | 20.48 | 7.60 |
| DoRA+NoRA | 6.80 | 1.09 | 20.94 | 7.60 |

Maximum recorded peak allocated CUDA memory across each method's three final seeds, including resident model components. These are full-task allocations, not adapter-only memory or reserved memory. Vision's recorded peak can include selected-checkpoint evaluation; extraction includes validation; retrieval and SDXL record training peaks. GPU memory scopes differ across tasks, so compare methods within a column.
