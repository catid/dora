# Shadow deletion diagnostics (v3)

These are training-side diagnostics on the fixed 64-example probe. They are not used for HPO selection. Each checkpoint uses exact token-progress replay at 20%, 50%, and 75% of the 512-update stream. ACT uses one shared effective score batch with at most 256 valid tokens. Adapter tensors and RNG state were restored and verified for every checkpoint.

| Method | Progress | Score | Spearman score/damage | Top-64 damage | Bottom-64 damage | Common-64 damage |
|---|---:|---|---:|---:|---:|---:|
| I_GATE | 0.20 | RAW | 0.344 | 0.007281 | 0.001643 | 0.002413 |
| I_GATE | 0.20 | GATE | 0.016 | 0.004512 | 0.002334 | 0.002413 |
| I_GATE | 0.20 | DISP | 0.310 | 0.007700 | 0.001780 | 0.002413 |
| I_GATE | 0.20 | ACT | -0.161 | 0.004444 | 0.002447 | 0.002413 |
| I_GATE | 0.50 | RAW | 0.128 | 0.010922 | 0.004802 | 0.004965 |
| I_GATE | 0.50 | GATE | 0.203 | 0.009103 | 0.004292 | 0.004965 |
| I_GATE | 0.50 | DISP | 0.184 | 0.010754 | 0.004735 | 0.004965 |
| I_GATE | 0.50 | ACT | -0.028 | 0.009840 | 0.008368 | 0.004965 |
| I_GATE | 0.75 | RAW | 0.253 | 0.012604 | 0.005513 | 0.007131 |
| I_GATE | 0.75 | GATE | 0.121 | 0.010431 | 0.006002 | 0.007131 |
| I_GATE | 0.75 | DISP | 0.296 | 0.012138 | 0.005787 | 0.007131 |
| I_GATE | 0.75 | ACT | 0.086 | 0.010338 | 0.010249 | 0.007131 |
| L_GATE | 0.20 | RAW | 0.159 | 0.013679 | 0.001094 | 0.001439 |
| L_GATE | 0.20 | GATE | 0.275 | 0.010712 | 0.000853 | 0.001439 |
| L_GATE | 0.20 | DISP | 0.127 | 0.013114 | 0.000825 | 0.001439 |
| L_GATE | 0.20 | ACT | 0.054 | 0.004569 | 0.001535 | 0.001439 |
| L_GATE | 0.50 | RAW | 0.054 | 0.019469 | 0.004067 | 0.003956 |
| L_GATE | 0.50 | GATE | -0.099 | 0.013112 | 0.002577 | 0.003956 |
| L_GATE | 0.50 | DISP | 0.054 | 0.018979 | 0.004240 | 0.003956 |
| L_GATE | 0.50 | ACT | 0.182 | 0.009306 | 0.006391 | 0.003956 |
| L_GATE | 0.75 | RAW | -0.030 | 0.017706 | 0.003895 | 0.004116 |
| L_GATE | 0.75 | GATE | -0.031 | 0.011767 | 0.003997 | 0.004116 |
| L_GATE | 0.75 | DISP | -0.036 | 0.017311 | 0.004144 | 0.004116 |
| L_GATE | 0.75 | ACT | 0.167 | 0.012259 | 0.008032 | 0.004116 |

The selected group damage is signed loss change; negative values would indicate that deletion improved the probe loss. Correlations are descriptive and use 64 single-slot deletions per checkpoint. No family winner is inferred from these probes.
