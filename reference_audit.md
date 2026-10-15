# Reference audit

The v2 implementation claims below are superseded where contradicted by the
[2026-10-15 protocol audit](reports/protocol_v2_audit.md). In particular v2 did
not train true displacement parameters, did not implement element-level AdaLoRA
score smoothing, and did not preserve its every-step final masking.

The corrected v3 implementation uses real U/V (and triplet displacements),
low-rank cancellation-free forward arithmetic, and a distinct reference path.
`src/rankstudy/v3/vendor/adalora` preserves original AdaLoRA source and its MIT
license at revision `d10f5ebee16c478fa2f41a44a237b38e8c9b0338`. Its RankAllocator
and compute_orth_regu execute without algorithm edits; outer factors and E use
the original initialization. A 40-step test matches parameters, element EMAs,
and final masking against upstream SVDLinear and RankAllocator.

Custom scoring uses every-update pre-optimizer task gradients, zero-start EMA
without bias correction, persistent monotonic masks and token-progress budgets.
Reference AdaLoRA keeps its upstream step-based schedule, threshold ties,
post-optimizer parameter/gradient pairing, between-event reactivation, and
final every-step mask projection. The nominal reference scale maps to alpha
= s * initial_rank; the actual multiplier with `ranknum + 1e-5` is recorded.
The faithful reference retains its small E-parameter budget mismatch. A future
budget-normalized variant must be separately registered.

The strict deterministic resume check exposed cuDNN backward nondeterminism.
The recorded native SDPA exception uses EFFICIENT_ATTENTION for all corrected
primary candidates after padded/causal FP64-oracle checks; actual-model profile
and resume remain validation gates. No attention mask is removed for speed.
No full paper reproduction is claimed.

## Historical v2 audit text (retained for provenance, not current validation)

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

The AdaLoRA arm follows the upstream update order: after each optimizer step it
updates `abs(parameter * gradient)` scores and their EMAs, then applies a mask
only at allocation events. A masked `E` coefficient is zeroed at that event but
is not forcibly zeroed before every later forward; Adam can therefore move it
and the slot can reactivate at a later allocation. The custom L/I allocators
remain pre-optimizer, monotonic, persistent-mask methods under the registered
protocol. An earlier launcher version zeroed AdaLoRA `E` before every forward;
it produced no terminal AdaLoRA records and was withdrawn before corrected
AdaLoRA HPO restarted.
