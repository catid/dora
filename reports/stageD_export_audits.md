# Stage-D separate-process export audits

Audited 33 exports from `/var/tmp/dora-adaptive-rank/runs/v3_stageD_export_audits_20261016_retry`.

The checker uses strict BF16 full-model logit tolerances. Structural checks and probe NLL are reported separately from BF16 rounding failures.

| Classification | Count |
|:--|--:|
| `bf16_logit_rounding_failure` | 33 |

| Run | Classification | Exact RMS | Exact max abs | Exact token NLL delta | Truncated token NLL delta | Factor max rel. error |
|:--|:--|--:|--:|--:|--:|--:|
| `COMPRESS_I_I_UNIFORM_SVD_s5001` | `bf16_logit_rounding_failure` | 0.0184019 | 0.5 | 0.000105823 | 0.0348974 | 3.8945e-08 |
| `COMPRESS_I_I_UNIFORM_SVD_s5002` | `bf16_logit_rounding_failure` | 0.0177199 | 0.335938 | -0.000102705 | 0.0349768 | 3.83826e-08 |
| `COMPRESS_I_I_UNIFORM_SVD_s5003` | `bf16_logit_rounding_failure` | 0.015851 | 0.625 | 0.000156081 | 0.0314688 | 3.85865e-08 |
| `COMPRESS_L_L_UNIFORM_SVD_s5001` | `bf16_logit_rounding_failure` | 0.0181119 | 0.4375 | 0.000136883 | 0.00318077 | 4.23525e-08 |
| `COMPRESS_L_L_UNIFORM_SVD_s5002` | `bf16_logit_rounding_failure` | 0.0157848 | 0.269531 | -0.000247854 | 0.00287409 | 4.51331e-08 |
| `COMPRESS_L_L_UNIFORM_SVD_s5003` | `bf16_logit_rounding_failure` | 0.0143598 | 0.25 | 2.94645e-05 | 0.00277361 | 4.41921e-08 |
| `FIXED_I_I_FIXED_s5001` | `bf16_logit_rounding_failure` | 0.0139159 | 0.3125 | -0.000301439 | 0.0213741 | 3.85951e-08 |
| `FIXED_I_I_FIXED_s5002` | `bf16_logit_rounding_failure` | 0.0161429 | 0.5 | 0.00017577 | 0.0208936 | 3.93298e-08 |
| `FIXED_I_I_FIXED_s5003` | `bf16_logit_rounding_failure` | 0.0173118 | 0.5 | 7.84218e-05 | 0.0198796 | 3.97725e-08 |
| `FIXED_L_L_FIXED_s5001` | `bf16_logit_rounding_failure` | 0.0151121 | 0.5 | 9.8587e-05 | 9.8587e-05 | 4.66448e-08 |
| `FIXED_L_L_FIXED_s5002` | `bf16_logit_rounding_failure` | 0.0175986 | 0.375 | 0.000114629 | 0.000114629 | 4.57674e-08 |
| `FIXED_L_L_FIXED_s5003` | `bf16_logit_rounding_failure` | 0.018663 | 0.314941 | 9.72298e-05 | 9.72298e-05 | 4.39472e-08 |
| `ONLINE_I_I_DISP_s5001` | `bf16_logit_rounding_failure` | 0.0161075 | 0.476562 | -0.00016393 | 0.0147851 | 4.82753e-08 |
| `ONLINE_I_I_DISP_s5002` | `bf16_logit_rounding_failure` | 0.0171085 | 0.375 | 0.000131689 | 0.0132889 | 4.4913e-08 |
| `ONLINE_I_I_DISP_s5003` | `bf16_logit_rounding_failure` | 0.0176481 | 0.433594 | 4.20854e-05 | 0.0115151 | 4.07261e-08 |
| `ONLINE_L_L_GATE_s5001` | `bf16_logit_rounding_failure` | 0.019495 | 0.4375 | -9.35177e-05 | -9.35177e-05 | 4.16071e-08 |
| `ONLINE_L_L_GATE_s5002` | `bf16_logit_rounding_failure` | 0.0154132 | 0.3125 | 3.58339e-05 | 3.58339e-05 | 5.01273e-08 |
| `ONLINE_L_L_GATE_s5003` | `bf16_logit_rounding_failure` | 0.0169806 | 0.453125 | 5.40559e-06 | 5.40559e-06 | 4.8921e-08 |
| `REFERENCE_ADALORA_REF_s5001` | `bf16_logit_rounding_failure` | 0.0144165 | 0.4375 | -0.000175599 | -0.000240962 | 4.1075e-08 |
| `REFERENCE_ADALORA_REF_s5002` | `bf16_logit_rounding_failure` | 0.0128 | 0.515625 | -9.53892e-05 | -9.53892e-05 | 4.50955e-08 |
| `REFERENCE_ADALORA_REF_s5003` | `bf16_logit_rounding_failure` | 0.015803 | 0.5 | 0.0001227 | -3.21071e-05 | 5.50812e-08 |
| `RESTART_I_I_PILOT_I_STUDENT_s5001` | `bf16_logit_rounding_failure` | 0.0172125 | 0.5 | 2.91026e-05 | 0.0109565 | 4.15302e-08 |
| `RESTART_I_I_PILOT_I_STUDENT_s5002` | `bf16_logit_rounding_failure` | 0.0154201 | 0.375 | -6.38783e-05 | 0.0109643 | 4.47916e-08 |
| `RESTART_I_I_PILOT_I_STUDENT_s5003` | `bf16_logit_rounding_failure` | 0.0104761 | 0.4375 | 5.07439e-05 | 0.00862022 | 4.76918e-08 |
| `RESTART_L_I_PILOT_L_STUDENT_s5001` | `bf16_logit_rounding_failure` | 0.0215204 | 0.328125 | -3.90395e-05 | -3.90395e-05 | 4.45197e-08 |
| `RESTART_L_I_PILOT_L_STUDENT_s5002` | `bf16_logit_rounding_failure` | 0.0182697 | 0.375 | -0.00013539 | -0.00013539 | 5.37857e-08 |
| `RESTART_L_I_PILOT_L_STUDENT_s5003` | `bf16_logit_rounding_failure` | 0.017388 | 0.375 | -0.000215647 | -0.000215647 | 5.54641e-08 |
| `TRIPLET_T0_T0_GATE_s5001` | `bf16_logit_rounding_failure` | 0.012378 | 0.4375 | -0.000115881 | -0.000115881 | 4.40246e-08 |
| `TRIPLET_T0_T0_GATE_s5002` | `bf16_logit_rounding_failure` | 0.0143798 | 0.410156 | 0.000118857 | 0.000118857 | 4.84469e-08 |
| `TRIPLET_T0_T0_GATE_s5003` | `bf16_logit_rounding_failure` | 0.01777 | 0.46875 | -9.39441e-06 | -9.39441e-06 | 4.35226e-08 |
| `TRIPLET_T1_T1_GATE_s5001` | `bf16_logit_rounding_failure` | 0.0114048 | 0.5 | -7.19054e-05 | 0.0135221 | 4.15018e-08 |
| `TRIPLET_T1_T1_GATE_s5002` | `bf16_logit_rounding_failure` | 0.019351 | 0.5 | 0.000275953 | 0.0119751 | 4.70943e-08 |
| `TRIPLET_T1_T1_GATE_s5003` | `bf16_logit_rounding_failure` | 0.0137237 | 0.515625 | 5.15063e-05 | 0.0133208 | 4.93041e-08 |
