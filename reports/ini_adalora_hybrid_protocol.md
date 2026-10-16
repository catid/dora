# Exploratory IniLoRA-alpha × AdaLoRA hybrid

This is a separately registered exploratory method. It is not included in the
41-cell primary tournament and does not replace `I_GATE` or `ADALORA_REF`.

The implementation is registered as `I_ADALORA_SVD` (fixed multiplier) and
`I_ADALORA_SVD_NORM` (multiplier divided by the current active rank plus
`1e-5`). A 16-update CUDA smoke run completed successfully on the pinned
Qwen2.5-Math-1.5B harness; it is an implementation check, not a quality
result. The smoke run reached the declared triplet budget (447 active slots
for a 448-slot-equivalent budget because each slot carries one amplitude) and
kept the full dormant tensors for possible reactivation.

## Representation

For each target matrix, use outer factors initialized independently as
`P0,Q0 ~ Normal(0, 0.5^2)` and an initial amplitude `lambda0=1`. Train
displacements `dP,dQ,dlambda`, initially zero:

```text
P = P0 + dP
Q = Q0 + dQ
lambda = lambda0 + dlambda
DeltaW = s [P diag(m * lambda) Q - P0 diag(m * lambda0) Q0]
```

The centered expansion is evaluated without subtracting two large products.
The mask applies to the complete centered triplet. The frozen original
backbone is never modified. This is the centered three-factor form closest to
combining IniLoRA's nonzero factor initialization with AdaLoRA's SVD-like
amplitude representation; it is not claimed to be the literal upstream
AdaLoRA initializer.

## AdaLoRA allocation component

Use the upstream-style element sensitivity for the active `P`, `Q`, and
`lambda` coordinates, with beta1=0.85 smoothing and beta2=0.85 uncertainty
weighting. Aggregate the three factor scores per component, use the declared
cubic warmup/shrink/final-freeze schedule, and allocate by global component
importance. The hybrid's mask/reactivation behavior is recorded separately
from the monotonic custom `T1_GATE` allocator.

Two scale variants must not be conflated:

1. Fixed multiplier `s`, matching the controlled centered-adapter policy.
2. AdaLoRA rank-number normalization `s/(r_active+1e-5)`, matching the
   faithful reference's numerical convention.

The primary hybrid candidate uses fixed `s`; the normalized version is a
registered scale ablation, since changing the multiplier during allocation is
itself a treatment.

The current exploratory allocator keeps the study-wide one-slot-per-module
floor and retains dormant factor storage. This avoids a singular denominator
in the rank-normalized ablation while still allowing reactivation among the
allocated slots. It is therefore distinct from the faithful AdaLoRA arm's
zero-rank eligibility. The hybrid score is the per-component sum of
AdaLoRA-style sensitivities for
the current `P`, `Q`, and `lambda`, followed by the same 0.85 mean and
uncertainty EMAs. Pruning zeros the current amplitude while masking the fixed
IniLoRA correction; the dormant amplitude remains trainable so a later event
can reactivate it.

## Controls and accounting

`T1_FIXED` and `T1_GATE` isolate centered triplet initialization and custom
allocation. `ADALORA_REF` isolates the faithful upstream parameterization and
allocator. Every hybrid candidate receives the same 12 LR/scale trials, two
full-horizon tuning seeds, and three fresh confirmation seeds. Selection uses
`dev_tune` only. Pilot, allocation, optimizer-rebuild, and any failed trial
costs are charged separately. No hybrid result is used to revise the primary
screening conclusions.
