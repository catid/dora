| Method | IID atom exact | OOD atom exact | Structural macro | Lexical macro | OOD strict exact | OOD valid syntax | OOD atom micro F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Task baseline | 9.38 | 8.18 | 0.00 | 9.55 | 7.74 | 56.85 | 13.12 |
| LoRA | 91.41 ± 2.07 | 67.01 ± 3.06 | 2.08 ± 1.80 | 77.84 ± 3.75 | 66.96 ± 3.10 | 96.92 ± 0.31 | 65.58 ± 1.91 |
| DoRA | 91.15 ± 3.16 | 66.47 ± 3.72 | 0.69 ± 1.20 | 77.43 ± 4.15 | 66.42 ± 3.64 | 96.63 ± 0.62 | 64.09 ± 2.83 |
| NoRA | 94.53 ± 0.78 | 70.59 ± 4.59 | 0.35 ± 0.60 | 82.29 ± 5.38 | 70.44 ± 4.73 | 97.62 ± 1.72 | 66.30 ± 3.03 |
| DoRA+NoRA | 91.93 ± 1.97 | 66.82 ± 4.39 | 1.39 ± 1.20 | 77.72 ± 5.12 | 66.77 ± 4.32 | 98.26 ± 0.17 | 64.04 ± 4.30 |
| DoRA+NoRA (slow magnitudes) | 92.71 ± 3.25 | 67.56 ± 4.84 | 2.08 ± 3.61 | 78.47 ± 5.28 | 67.21 ± 4.63 | 95.93 ± 2.54 | 66.58 ± 4.04 |
| DoRA+NoRA (input gains) | 92.45 ± 0.45 | 67.36 ± 2.68 | 0.69 ± 0.60 | 78.47 ± 3.03 | 67.26 ± 2.85 | 98.21 ± 0.45 | 64.65 ± 2.38 |

COGS structural and lexical results remain separate: 18 lexical categories and 3 structural categories. Category sample sizes are equal. IID is a separate 128-example sample. Values are percentages; variability is across training seeds.

The executed COGS runner recorded `hit_generation_cap`/`generation_cap_hits` by checking whether tokenizer EOS 151645 was absent. The pinned generation configuration also stops on EOS 151643, so these flags can falsely indicate token-limit exhaustion. Raw generated token IDs were not saved, preventing an exact retrospective cap audit. Saved text and its atom-set exact match, strict exact match, validity and F1 scores are unaffected. The archive retains the executed source; later EOS-detection or raw-token-logging fixes apply only to future runs and do not change these records.
