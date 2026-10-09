| Task | Rank | Method | Recall@10 ×100 | MRR@10 ×100 | nDCG@10 ×100 |
|:--|--:|:--|--:|--:|--:|
| nfcorpus | 2 | Task baseline | 15.50 | 50.60 | 31.68 |
| nfcorpus | 2 | LoRA | 15.43 ± 0.13 | 50.62 ± 0.78 | 31.98 ± 0.23 |
| nfcorpus | 2 | DoRA | 15.52 ± 0.09 | 50.56 ± 0.68 | 32.01 ± 0.21 |
| nfcorpus | 2 | NoRA | 15.48 ± 0.14 | 50.09 ± 0.74 | 31.90 ± 0.65 |
| nfcorpus | 2 | DoRA+NoRA | 15.56 ± 0.19 | 50.18 ± 0.45 | 31.92 ± 0.55 |
| nfcorpus | 2 | DoRA+NoRA (slow magnitudes) | 15.58 ± 0.17 | 49.95 ± 0.61 | 31.89 ± 0.57 |
| nfcorpus | 2 | DoRA+NoRA (input gains) | 15.58 ± 0.20 | 49.96 ± 0.58 | 31.87 ± 0.50 |
| nfcorpus | 8 | Task baseline | 15.50 | 50.60 | 31.68 |
| nfcorpus | 8 | LoRA | 15.80 ± 0.31 | 50.84 ± 0.43 | 32.20 ± 0.20 |
| nfcorpus | 8 | DoRA | 15.80 ± 0.30 | 50.95 ± 0.40 | 32.24 ± 0.15 |
| nfcorpus | 8 | NoRA | 15.95 ± 0.13 | 50.67 ± 0.42 | 32.17 ± 0.30 |
| nfcorpus | 8 | DoRA+NoRA | 15.92 ± 0.21 | 50.64 ± 0.47 | 32.17 ± 0.38 |
| nfcorpus | 8 | DoRA+NoRA (slow magnitudes) | 15.91 ± 0.17 | 50.84 ± 0.45 | 32.20 ± 0.40 |
| nfcorpus | 8 | DoRA+NoRA (input gains) | 15.19 ± 0.05 | 49.32 ± 1.52 | 31.31 ± 0.46 |
| scifact | 2 | Task baseline | 78.83 | 60.51 | 64.63 |
| scifact | 2 | LoRA | 74.87 ± 0.59 | 57.50 ± 0.18 | 61.12 ± 0.11 |
| scifact | 2 | DoRA | 74.89 ± 0.40 | 57.84 ± 0.11 | 61.38 ± 0.25 |
| scifact | 2 | NoRA | 72.81 ± 1.33 | 56.70 ± 0.66 | 59.99 ± 0.83 |
| scifact | 2 | DoRA+NoRA | 73.03 ± 1.22 | 56.82 ± 0.88 | 60.12 ± 0.96 |
| scifact | 2 | DoRA+NoRA (slow magnitudes) | 73.14 ± 1.07 | 56.83 ± 0.84 | 60.14 ± 0.89 |
| scifact | 2 | DoRA+NoRA (input gains) | 73.00 ± 1.23 | 56.70 ± 0.82 | 60.02 ± 0.94 |
| scifact | 8 | Task baseline | 78.83 | 60.51 | 64.63 |
| scifact | 8 | LoRA | 75.70 ± 0.21 | 59.31 ± 0.87 | 62.64 ± 0.60 |
| scifact | 8 | DoRA | 75.79 ± 0.17 | 59.38 ± 0.80 | 62.66 ± 0.59 |
| scifact | 8 | NoRA | 74.65 ± 0.92 | 58.99 ± 0.64 | 62.11 ± 0.52 |
| scifact | 8 | DoRA+NoRA | 74.67 ± 0.96 | 59.03 ± 0.51 | 62.12 ± 0.34 |
| scifact | 8 | DoRA+NoRA (slow magnitudes) | 74.61 ± 0.86 | 59.07 ± 0.69 | 62.14 ± 0.49 |
| scifact | 8 | DoRA+NoRA (input gains) | 70.49 ± 0.88 | 55.08 ± 0.97 | 58.28 ± 0.51 |
