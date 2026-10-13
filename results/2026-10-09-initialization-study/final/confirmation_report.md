# Fresh-seed initialization confirmation

Configurations were selected using the complete seed-42 validation grid. These results use only fresh training seeds 45–49 and are separate from the primary seeds 42–44. Roles sharing a candidate reuse its fit; they do not add independent replicates. No configuration is selected by confirmation test scores. A failed seed makes the five-seed mean and its paired contrast unavailable.

Metrics are nDCG ×100, Aircraft macro accuracy percent, and ViGGO JSON exact match percent. Means use five seeds; SD is the sample SD. Exploratory 95% intervals use 10,000 crossed paired seed/example bootstrap draws, with class stratification/equal class weights for Aircraft and no multiplicity correction. These are fresh training seeds on previously used held-out examples, not a fresh test dataset.

| Dataset | Role | Candidate | Mean ± SD |
|---|---|---|---:|
| scifact | validation_winner | inilora_alpha__i3__lr1 | 70.37 ± 1.34 |
| scifact | accepted_anchor | accepted_anchor__i0__lr1 | 68.60 ± 1.39 |
| scifact | accepted_gaussian_scale | accepted_gaussian_scale__i3__lr1 | 69.63 ± 1.47 |
| nfcorpus | validation_winner | student_t3_up__i2__lr1 | 32.05 ± 0.40 |
| nfcorpus | accepted_anchor | accepted_anchor__i0__lr1 | 31.92 ± 0.47 |
| nfcorpus | accepted_gaussian_scale | accepted_gaussian_scale__i3__lr1 | 31.87 ± 0.54 |
| scifact_transfer | validation_winner | student_t3_up__i2__lr1 | 64.33 ± 0.90 |
| scifact_transfer | accepted_anchor | accepted_anchor__i0__lr1 | 63.72 ± 0.75 |
| scifact_transfer | accepted_gaussian_scale | accepted_gaussian_scale__i3__lr1 | 62.11 ± 1.36 |
| extraction | validation_winner | truncnormal2_down__i3__lr1 | 76.02 ± 0.65 |
| extraction | accepted_anchor | accepted_anchor__i0__lr1 | 68.52 ± 5.33 |
| extraction | accepted_gaussian_scale | accepted_gaussian_scale__i3__lr0 | 69.38 ± 2.74 |
| aircraft | validation_winner | loraga_ArBr__i1__lr3 | 62.04 ± 1.16 |
| aircraft | accepted_anchor | accepted_anchor__i0__lr3 | 55.42 ± 1.54 |
| aircraft | accepted_gaussian_scale | accepted_gaussian_scale__i1__lr3 | 56.59 ± 1.03 |

| Dataset | Winner comparison | Difference [95% CI] |
|---|---|---:|
| scifact | accepted_anchor | +1.77 [-0.55, +4.23] |
| scifact | accepted_gaussian_scale | +0.73 [-1.42, +2.95] |
| nfcorpus | accepted_anchor | +0.12 [-0.68, +0.91] |
| nfcorpus | accepted_gaussian_scale | +0.18 [-0.70, +1.05] |
| scifact_transfer | accepted_anchor | +0.61 [-1.09, +2.13] |
| scifact_transfer | accepted_gaussian_scale | +2.22 [+0.01, +4.30] |
| extraction | accepted_anchor | +7.50 [+1.25, +13.52] |
| extraction | accepted_gaussian_scale | +6.64 [+2.27, +11.09] |
| aircraft | accepted_anchor | +6.62 [+5.45, +7.77] |
| aircraft | accepted_gaussian_scale | +5.44 [+4.41, +6.48] |
