# V3 validation evidence

| Measured conclusion | Evidence and limits |
|---|---|
| All 41/41 registry cells execute on Qwen1.5B | 32 updates, identical input/supervised exposure and original backbone; engineering configurations only. |
| Common export budget fixed | 4 new global-energy exports each contain 1,376,256 ordinary-factor scalars / 5,505,024 factor bytes; original underspending runs retained. |
| Interrupted pipelines reproduce controls | Restart student and compression recovery match final tensors, Adam, score state, RNG and exposure bit for bit. |
| Exact export algebra passes FP32 model oracles | All five representation families pass; ordinary exports load on the original backbone in separate processes. |
| Conservative BF16 export parity fails for four cases | Update reconstruction and FP32 checks pass, but BF16 rounding changes full-model logits; raw distortion remains reported, not relabeled exact numerical equality. |
| DDP matches at identical checkpoints | Four real-model cases × four updates pass through no_sync accumulation, unequal token counts, pruning and DDP/optimizer rebuild. |
| Independent DDP trajectories do not meet strict tolerances | BF16 gradient and FP32 parameter checks failed at update two; free-trajectory equivalence remains unproven. Primary runs stay on one GPU. |

## Export checks

The presealed 64-example training probe is used in every case. FP32 rows promote the same stored BF16 weight values solely as an arithmetic oracle; they are not primary training/evaluation results. The exact refactor may exceed the final budget. Truncated NLL differences measure immediate distortion before recovery. No development or test selection uses these probes.

| Method | Arithmetic | Strict result | Exact logit relative RMS | Exact NLL change | Truncated NLL change |
|---|---|---|---:|---:|---:|
| L_FIXED | bf16 | failed | 0.00818 | 0.000257 | 0.000257 |
| L_PILOT_I_STUDENT | bf16 | failed | 0.011 | -4.38e-06 | 0.00457 |
| T0_GATE | bf16 | failed | 0.0121 | 0.000149 | 0.000243 |
| T1_GATE_ORTHO | bf16 | failed | 0.00967 | 5.85e-05 | 0.00373 |
| L_FIXED | fp32_oracle | passed | 9.76e-07 | -1.75e-08 | -1.75e-08 |
| L_PILOT_I_STUDENT | fp32_oracle | passed | 1.08e-06 | 2.18e-08 | 0.00478 |
| T0_GATE | fp32_oracle | passed | 1.01e-06 | 1.16e-09 | 0.00026 |
| T1_GATE_ORTHO | fp32_oracle | passed | 1.03e-06 | 3.33e-08 | 0.00344 |
| ADALORA_REF | bf16 | passed | 0 | 0 | -0.000401 |
| ADALORA_REF | fp32_oracle | passed | 9.71e-07 | 4.89e-09 | 1.03e-06 |

All thresholds and failed records remain preserved. Passing the FP32 oracle does not erase the BF16 failure. DDP comparisons at identical states do not prove independent trajectories remain identical. Neither diagnostic establishes a quality winner.

Primary tuning, independent screening, confirmation, equal-compute/export quality comparisons, 7B and official benchmark evaluations remain pending. Generation/parser validation and shadow deletion/curvature/gauge/restoration work remain open.

Commands: `scripts/check_export_v3.py`, `scripts/check_llm_ddp_v3.py`, `scripts/smoke_registry_v3.py`; exact invocations are stored with each raw run.
