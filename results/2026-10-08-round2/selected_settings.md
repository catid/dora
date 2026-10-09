| Task | Rank | Method | Base LR | Magnitude LR multiplier | Selected epoch/step |
|:--|--:|:--|--:|--:|:--|
| aircraft | 2 | Task baseline | 0.001 | 1 | [15, 18, 18] |
| aircraft | 2 | LoRA | 0.001 | 1 | [14, 19, 19] |
| aircraft | 2 | DoRA | 0.001 | 1 | [18, 18, 19] |
| aircraft | 2 | NoRA | 0.0003 | 1 | [19, 20, 20] |
| aircraft | 2 | DoRA+NoRA | 0.0003 | 1 | [19, 19, 20] |
| aircraft | 2 | DoRA+NoRA (slow magnitudes) | 0.001 | 0.1 | [20, 20, 18] |
| aircraft | 2 | DoRA+NoRA (input gains) | 0.0003 | 1 | [20, 19, 19] |
| aircraft | 8 | Task baseline | 0.001 | 1 | [15, 18, 18] |
| aircraft | 8 | LoRA | 0.001 | 1 | [18, 19, 19] |
| aircraft | 8 | DoRA | 0.001 | 1 | [18, 16, 20] |
| aircraft | 8 | NoRA | 0.0003 | 1 | [18, 20, 19] |
| aircraft | 8 | DoRA+NoRA | 0.0003 | 1 | [17, 20, 17] |
| aircraft | 8 | DoRA+NoRA (slow magnitudes) | 0.001 | 0.01 | [20, 20, 20] |
| aircraft | 8 | DoRA+NoRA (input gains) | 0.0003 | 1 | [17, 20, 20] |
| nfcorpus | 2 | Task baseline | — | — | — |
| nfcorpus | 2 | LoRA | 0.001 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 2 | DoRA | 0.001 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 2 | NoRA | 0.0003 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 2 | DoRA+NoRA | 0.0003 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 2 | DoRA+NoRA (slow magnitudes) | 0.0003 | 0.1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 2 | DoRA+NoRA (input gains) | 0.0003 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 8 | Task baseline | — | — | — |
| nfcorpus | 8 | LoRA | 0.0003 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 8 | DoRA | 0.0003 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 8 | NoRA | 0.0001 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 8 | DoRA+NoRA | 0.0001 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 8 | DoRA+NoRA (slow magnitudes) | 0.0001 | 0.01 | ['final epoch', 'final epoch', 'final epoch'] |
| nfcorpus | 8 | DoRA+NoRA (input gains) | 0.0003 | 1 | ['final epoch', 'final epoch', 'final epoch'] |
| cogs | 8 | Task baseline | — | — | — |
| cogs | 8 | LoRA | 0.0003 | 1 | [384, 384, 384] |
| cogs | 8 | DoRA | 0.0003 | 1 | [288, 384, 288] |
| cogs | 8 | NoRA | 0.0001 | 1 | [384, 384, 384] |
| cogs | 8 | DoRA+NoRA | 0.0001 | 1 | [384, 384, 288] |
| cogs | 8 | DoRA+NoRA (slow magnitudes) | 0.0003 | 0.1 | [384, 288, 384] |
| cogs | 8 | DoRA+NoRA (input gains) | 0.0001 | 1 | [384, 288, 288] |
