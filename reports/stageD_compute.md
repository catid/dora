# Stage-D equal charged-GPU-second comparison

The retry queue contains 66 jobs with a common cap of 388.1408279319294 charged GPU-seconds.
The first one-second-reserve attempt remains preserved separately and is excluded.

| Classification | Count |
|:--|--:|
| `complete` | 1 |
| `infeasible_pilot_budget` | 12 |
| `infeasible_recovery_budget` | 3 |
| `infeasible_source_budget` | 50 |

| Method | Runs | Complete | Infeasible | Complete NLL mean |
|:--|--:|--:|--:|--:|
| `ADALORA_REF` | 6 | 0 | 6 |  |
| `I_DISP` | 6 | 0 | 6 |  |
| `I_FIXED` | 6 | 0 | 6 |  |
| `I_PILOT_I_STUDENT` | 6 | 0 | 6 |  |
| `I_PILOT_L_STUDENT` | 6 | 0 | 6 |  |
| `I_UNIFORM_SVD` | 6 | 0 | 6 |  |
| `L_FIXED` | 6 | 0 | 6 |  |
| `L_GATE` | 6 | 1 | 5 | 0.196847 |
| `L_UNIFORM_SVD` | 6 | 0 | 6 |  |
| `T0_GATE` | 6 | 0 | 6 |  |
| `T1_GATE` | 6 | 0 | 6 |  |

A complete row finished its source, transition, and required ordinary-LoRA recovery within the cap. Infeasible rows retain their partial checkpoints and measured charged work; they are not quality scores.
