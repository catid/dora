Aircraft: all 3,333 validation images, trained seed-42 baseline classifier, zero-update adapters. The FP32 control disables TF32; BF16 uses the training precision protocol. The shared trained classifier probes backbone sensitivity; it does not replay each adapter training run's initially random classifier. These are initialization diagnostics, not trained-adapter test results.

| Untrained backbone adapter | FP32 max logit drift from bare | BF16 max logit drift from bare | BF16 changed predictions | BF16 correct |
|:--|--:|--:|--:|--:|
| Bare frozen backbone | 0 | 0 | 0/3333 | 1451/3333 |
| LoRA | 1.93119e-05 | 0.210938 | 69/3333 | 1446/3333 |
| DoRA | 5.36442e-05 | 0.218262 | 63/3333 | 1443/3333 |
| NoRA | 1.93119e-05 | 0.210938 | 69/3333 | 1446/3333 |
| DoRA+NoRA | 5.36442e-05 | 0.218262 | 63/3333 | 1443/3333 |
| DoRA+NoRA (slow magnitudes) | 5.36442e-05 | 0.218262 | 63/3333 | 1443/3333 |
| DoRA+NoRA (input gains) | 5.36442e-05 | 0.218262 | 63/3333 | 1443/3333 |

Ranks 2 and 8 are bitwise equal within each method in this zero-update control. All FP32 adapter cases have 0 changed predictions relative to bare (maximum logit drift 5.36442e-05). The bare BF16 result exactly matches the saved baseline validation record (1451/3333 correct).

LoRA and NoRA form one bitwise-equal BF16 family; the four DoRA-based methods form another. Between these families, 43/3333 predictions change, with mean absolute logit drift 0.0200346 and maximum 0.164062. The families differ by 3 correct predictions overall. Disagreement counts measure prediction changes; they do not establish an accuracy loss of equal size or predict effects after training. Aircraft therefore does not have numerically identical BF16 initialization across all six methods. Tiny trained-method differences should be interpreted with this qualification.

A separate mechanism probe on the first 128 validation images recalibrates a newly constructed rank-8 DoRA instance using GPU weight norms. Its maximum magnitude/norm ratio error drops from 2.38419e-07 to 0; its logits then match LoRA bitwise in both FP32 and BF16. This supports CPU/GPU norm reduction rounding as the source of the family difference in this probe. The recalibrated adapter retains the separate projection/bias numerical path: relative to bare BF16 it still changes 3/128 predictions, with maximum logit drift 0.179688. The probe does not modify any trained adapter, original control or reported task result.

The original 128-image Aircraft control is also retained and independently re-scored. It covers the first four classes in manifest order, so its accuracy is not representative of the full validation split. Both controls retain all 26 logit arrays and 132 paired comparisons; the full control is the primary numerical diagnostic.

Retrieval initialization control:

| Untrained model | NFCorpus nDCG@10 ×100 | SciFact nDCG@10 ×100 | Max probe embedding difference |
|:--|--:|--:|--:|
| Native frozen | 31.6850 | 64.6321 | 0.0000000 |
| LoRA | 31.7529 | 64.5386 | 0.0013143 |
| DoRA | 31.7529 | 64.5386 | 0.0013143 |
| NoRA | 31.7529 | 64.5386 | 0.0013143 |
| DoRA+NoRA | 31.7529 | 64.5386 | 0.0013143 |
| DoRA+NoRA (slow magnitudes) | 31.7529 | 64.5386 | 0.0013143 |
| DoRA+NoRA (input gains) | 31.7529 | 64.5386 | 0.0013143 |

Zero-update adapters split BF16 projection and bias addition while native Linear may fuse them. This changes rounding slightly. All six untrained adapter methods produced identical measured retrieval metrics; the control uses rank 2/seed 42 and 128 fixed document embeddings. Method-to-method comparisons share this numerical path. Native frozen baselines are retained visibly in the main tables.
