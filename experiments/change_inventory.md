# Scope of the DoRA change ablations

The current default is the original source plus FP32 magnitude storage, corresponding to `ab_m_fp32`. The revised implementation discussed below is now preserved as `dora_row.py`; archived sources and results retain their historical names.

The reference is the original implementation at `bb97617a0d5e1ad4c6856cb7278f5a7386820d18`. The published revision is `9d102e3e9d6320bd251086aa17afef665953f602`. The downstream study compares the **task adapters** that produced the reported scores. Its revised endpoint is `experiments.adapters.AdapterLinear(method="dora")`; this is not identical to the preserved revised `dora_row.DoRALayer` constructor.

The twelve recipes test six single additions to the original and a cumulative path to the revised task recipe. Each task uses three seeds and two analysis lanes: the original-selected learning rate for all recipes, and an independent equal-budget validation search for each recipe. A shared fit is counted once when both lanes select the same rate. Individual effects and successive cumulative increments answer different questions and must not be added together as independent contributions.

| Change | Intervention in the downstream matrix | Interpretation |
|:--|:--|:--|
| Normalization axis | Input columns (`dim=0`) to output rows (`dim=1`) | Changes magnitude layout and, for nonsquare weights, parameter count. This is an algorithm/capacity choice. |
| Denominator gradient | Full derivative to detached denominator | Optional memory-saving approximation, not a necessary repair of the original full-gradient formula. |
| Magnitude storage | Base dtype to FP32 | Casts the original initial norm after computing it. Does not silently change how the initial norm is computed. Already a no-op for the FP32 Aircraft base. |
| Norm arithmetic | Native/ambient arithmetic to FP32 outside autocast | Changes initial and denominator norm work. Dense numerator arithmetic stays historical. FP64 reference arithmetic is retained. |
| Factor initializer | Historical normal up-factor/zero down-factor to Kaiming down-factor/zero up-factor | A **task-harness change**, not the standalone repository initializer. Includes orientation, distribution, RNG device, and parameter registration order used in those measured endpoints. |
| Projection | Dense adapted weight to separate base/low-rank projections | Algebraically equivalent at fixed other settings in real arithmetic; changes association, rounding, and sometimes arithmetic dtype under BF16. |

The revised endpoint also contains a tiny-value norm floor. It is tracked separately: any floor activation rejects a run from the six-change attribution. Its zero-weight behavior belongs in a targeted correctness test, not a fabricated task-quality score when the guard is inactive on pretrained weights.

Some edits concern API behavior or the demo and are not active interventions in these pretrained task loops:

`audit_engineering_changes.py` reproduces six concrete boundaries against the literal archived source. It confirms the biasless-conversion exception, zero-weight NaNs, mixed-BF16 projection exception, and import-time RNG reset, and checks revised standalone initialization and root/shared-module conversion. Its saved evidence includes both source versions. Root conversion is labeled an API expansion; these checks are not a downstream ranking.

| Edit | Concrete check or evidence | Downstream scope |
|:--|:--|:--|
| Initialize a newly constructed standalone base weight | `test_fresh_layer_has_initialized_weight_and_no_bias` | Tasks load pretrained weights; this branch is inactive. |
| Preserve absent bias and permit biasless conversion | `test_initial_conversion_preserves_linear_for_all_input_shapes`, original-adapter parity checks | Necessary compatibility for biasless pretrained projections; absent bias stays absent in every recipe. |
| Preserve device/dtype; validate dimensions and rank | `test_cpu_dtype_is_preserved`, `test_cuda_device_dtype_forward_and_backward`, `test_invalid_dimensions_rank_and_weight_are_rejected` | Valid inputs and common placement are held fixed. Standalone factor dtype is an additional numerical policy, discussed below. |
| Stop resetting the caller's RNG during import | `test_import_does_not_reset_random_generator`, original-adapter import/RNG checks | Each task deliberately seeds immediately before constructing its adapters. |
| Copy base storage and preserve shared module aliases | `test_frozen_base_is_independent_of_source_storage`, `test_replacement_handles_root_nested_and_shared_layers` | Task backends intentionally share the frozen pretrained storage and verify its digest after training. |
| Merge adapters for inference | `test_merge_matches_adapted_forward_and_is_independent` | Training/evaluation in this study uses live adapters, not merged weights. |
| Initialize/floor zero rows and protect half-precision reductions/rescaling | Zero-row, tiny-weight and FP16 overflow tests in `test_dora_row.py` | Nonzero norm diagnostics establish whether the floor is inactive; FP32 arithmetic changes are swept separately. |
| Evaluate the demo on a fixed held-out set | Archived demo and toy diagnostic | Makes the demo score interpretable, but does not turn its one-output problem into a valid general method ranking. |

The standalone revised constructor keeps the historical normal up-factor/zero down-factor orientation, but draws its factors directly in the base dtype/device. The task harness instead uses FP32 factors for both measured endpoints and uses Kaiming initialization for the revised endpoint. Consequently, these six ablations must not be described as an exhaustive measurement of every standalone numerical default. In particular, standalone BF16 factor storage is not the same intervention as BF16 magnitude storage.

Three separately audited SciFact supplements close parts of this boundary without changing the main twelve recipes:

- [Factor storage](../results/2026-10-09-factor-storage/quality_table.md) casts identical FP32-initialized factors to BF16. Neither endpoint's storage-only effect is resolved by its exploratory interval. This includes effects on gradient and optimizer precision, and on the original dense update's implicit arithmetic dtype; it does not test native BF16 random draws in isolation.
- [LoRA initialization](../results/2026-10-09-lora-initializer/quality_table.md) compares conventional and historical initializer recipes in the same LoRA forward. Historical initialization raises mean SciFact nDCG×100 from 65.10 to 67.97. Original DoRA is 67.90; their difference is unresolved. Matching initialization still does not isolate normalization because magnitude parameters and projection association differ.
- [Actual standalone class](../results/2026-10-09-standalone-endpoint/quality_table.md) called the then-current `dora.DoRALayer.from_linear` directly (now preserved byte-for-byte as `dora_row.DoRALayer.from_linear`). It scores 68.23 ± 1.25 versus the original's 67.90 ± 0.60; the difference is unresolved. It does not inherit the revised **task adapter** score of 65.38. This is a complete-recipe endpoint comparison, not a factorial decomposition of native dtype/RNG changes. Monitoring hooks repeat dense norm work, so instrumented timing is not a production speed measurement.

Endpoint parity is a prerequisite, not evidence of task superiority. The study verifies original/revised initialization, RNG consumption, parameter ordering, nonzero outputs and gradients, and clipped AdamW trajectories. A separate full Qwen/ViGGO check reproduces five updates and all 216 saved adapter tensors bitwise for each endpoint. Completed full-budget endpoint controls additionally reproduce all historical pilots/finals, checkpoint tensors and saved predictions across retrieval, Aircraft and ViGGO. These controls are retained in the [endpoint reproduction report](../results/2026-10-09-change-ablations/endpoint_reproduction.md). The ablation tasks were selected after observing endpoint disagreements, so their results are exploratory explanations on these test sets rather than a fresh confirmatory benchmark.
