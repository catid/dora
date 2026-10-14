# Reference audit

## Faithful behavior

IniLoRA-alpha's inspected implementation (`R4`) draws both LoRA factors independently from `Normal(0, 0.5)` and subtracts the scaled initial product from the frozen weight. It does not perform the paper's optional weight-approximation optimization in this controlled recipe.

AdaLoRA's reference implementation (`R6`) uses outer factors `A,B ~ Normal(0, 0.02)`, a zero middle coefficient `E`, and forward update `B @ (A * E) * scaling / (ranknum + 1e-5)`. Its rank allocator smooths elementwise `abs(parameter * gradient)` with beta1, multiplies by an uncertainty EMA with beta2, combines A/E/B triplet scores, uses a cubic budget schedule, and masks low-scoring triplets. The original reference adds an unnormalized Frobenius orthogonality term (the common example coefficient is 0.1). The pinned PEFT source (`R7`) exposes the same parameterization but its `reset_lora_parameters` path initializes `E` with a nonzero Gaussian; the preregistration explicitly requires zero middle coefficients, so this study uses the original/reference zero-E behavior and records that compatibility patch rather than silently calling the two initializers identical.

## Mathematical equivalents

Our centered adapters train displacements `U=A-A0`, `V=B-B0` and evaluate `B0@U + V@A0 + V@U`; this is algebraically identical to `BA-B0A0` and avoids cancellation. FP32 adapter arithmetic is used initially while the frozen backbone remains BF16. This is a numerical implementation, not a changed method.

## Controlled custom variants

Custom `L` and `I` two-factor adapters use the same fixed forward multiplier and masks. RAW, GATE, DISP, and ACT are slot-level estimators with EMA beta 0.85. Custom pruning is monotonic with a persistent mask and minimum-rank floor. Three-factor T0/T1, rank-pattern restart, and train-compress-recover are explicitly separate registered methods.

## Compatibility differences

The faithful AdaLoRA reference is kept as a distinct arm because its initialization, element-level score aggregation, uncertainty weighting, rank-number denominator offset, orthogonality penalty, and masking/re-activation semantics differ from the custom methods. The current environment uses the 2026-10-13 PyTorch nightly rather than the older dependency set in upstream repositories. Native SDPA is used; backend selection is recorded at runtime. No claim of a full paper reproduction is made: this is a controlled minimal arm with the required q/o target modules and task loop.
