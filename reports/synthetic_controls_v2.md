# Frozen-matrix Stage A control

This report is generated from `artifacts/synthetic_controls_v2.json`. It uses two seeds, eight independent 16×16 modules, teacher ranks `[0,0,1,1,2,4,8,16]`, 100 Adam steps, slot budgets 8/16/32, isotropic and anisotropic Gaussian inputs, and clean/noisy targets.

Values are mean MSE over the eight modules and the two seeds in each cell; this is a mechanistic control, not an LLM quality result.

| Init | Anisotropic | Noise | Budget | Mean MSE |
|---|---:|---:|---:|---:|
| I | false | false | 8 | 16.496493 |
| I | false | false | 16 | 11.885868 |
| I | false | false | 32 | 7.014073 |
| I | false | true | 8 | 22.804765 |
| I | false | true | 16 | 17.001921 |
| I | false | true | 32 | 8.758780 |
| I | true | false | 8 | 50.250845 |
| I | true | false | 16 | 36.371811 |
| I | true | false | 32 | 20.684177 |
| I | true | true | 8 | 66.330162 |
| I | true | true | 16 | 50.233552 |
| I | true | true | 32 | 27.290665 |
| L | false | false | 8 | 19.712187 |
| L | false | false | 16 | 15.244005 |
| L | false | false | 32 | 8.944201 |
| L | false | true | 8 | 23.542305 |
| L | false | true | 16 | 17.894550 |
| L | false | true | 32 | 10.234597 |
| L | true | false | 8 | 70.196955 |
| L | true | false | 16 | 51.870411 |
| L | true | false | 32 | 31.466718 |
| L | true | true | 8 | 71.898092 |
| L | true | true | 16 | 54.447959 |
| L | true | true | 32 | 33.992971 |
