# Initialization study protocol, 2026-10-09

This is an exploratory continuation of the accepted DoRA ablations. The public
`dora.py` is not changed. Its SHA256 remains
`5f11aa1c64b6f02fb6bb8fb4558c45b12e1501f4543c40afb654bfc5b6958f24`.
It retains the original implementation's input-column weight normalization,
full denominator gradients, dense projection, implicit adapter scale1, original
normal-up/zero-down initialization, and only the accepted FP32 magnitude-storage
change (FP64 preserved). No experiment below silently becomes the public default.

## Scope and frozen candidate opportunity

`init_registry.py` is the executable candidate definition. JSON registries are
written once, bind this protocol's exact hash, and enumerate every initializer
configuration and learning rate before any main-study validation selection.
Main centered study: 76 families, comprising the single-setting exact anchor
and 75 families with four initializer settings. Warm-start study: exact anchor
plus five four-setting families. Quantization study: exact anchor, four
single-setting paired quantizer controls, and four four-setting LoftQ families.
Every setting is crossed with the task's complete original LR grid. A
single-setting anchor/control is never described as receiving equal initializer
tuning. `accepted_gaussian_scale` supplies the equal-budget original-draw family.

Parent-family candidate totals are reported alongside child-family totals.
For example, Gaussian orientation/RMS/scale controls, multiple Student-t dfs,
CorDA IPM/KPM, and LoRA-GA directions receive more aggregate attempts than one
child family; renaming children does not conceal this expanded opportunity.
Fixed-original-LR selection searches initialization only. Joint selection searches
initialization and LR. All registered pilots for a task and study lane must
finish before its selections freeze and before its test evaluation starts.
No candidate is added or removed because of validation or test performance.

The user-authorized runtime budget is up to three days on four local RTX PRO
6000 Blackwell Max-Q GPUs. The main grid was sized for approximately24hours,
subject to measured decomposition/prefit cost. Additional time supports the
registered slow recipes, audits and a validation-selected finalist check on fresh
training seeds. These are estimates, not automatic study-completion conditions. Interruptions preserve
partial status; unfinished cells are never reported as completed or negative
results. Fixtures are infrastructure evidence, separate from the main study.

## Initialization interventions

Canonical factors are up B[out,r], down A[r,in]; experimental parameter names
are lora_B and lora_A respectively (public names are reversed). Controlled random
initializers first consume the accepted global CPU draws, then replace values
using a local seed bound to fit seed, module path and orientation. This preserves
downstream batching, dropout and classifier initialization. Both factors remain
trainable, parameter registration and optimizer groups remain unchanged.

Noise shapes: Gaussian, uniform, Rademacher, Laplace, Student-t(df3/5/10),
variance-corrected truncated normal(bounds1/2), sparse Rademacher(p.01/.1/.5),
QR orthogonal, DCT and cropped Hadamard, in both one-zero orientations.
Ordinary shape comparisons use analytic entry RMS gain/sqrt(rank), with gains
.25/.5/1/2. Explicit controls test sample-RMS scaling, spherical rank vectors,
fan-in scaling, standard LoRA Kaiming uniform, repeated-identity BIMI and signed
permuted BIMI. Cropped Hadamard uses the next-power-of-two Sylvester basis and
does not claim exact column orthogonality after cropping. Sparse zero vectors
are recorded, never silently repaired.

RBF, exponential and Matern3/2 rank-coordinate covariance use unit diagonal,
gain1 and lengths .25/.5/1/2 in each orientation. Rank coordinates have no semantic
spatial order. These test covariance conditioning, not neuron proximity.
One-time spherical/BIMI initialization is distinct from per-forward NoRA.
PEFT complementary-subspace factors, nonzero LoRA's both-uniform original-input
scaling and IniLoRA alpha/beta add both-nonzero random controls with explicitly
centered products.

Primary-source inventory and exact formula boundaries are in
`initialization_literature.md`. Executable factor families are PiSSA, MiLoRA,
OLoRA, EVA, CorDA, Astra, LoRA-GA, LoRA-One, FILet, LoRA-DA, CNTLoRA, IniLoRA,
TLoRA and MiCA. Four-setting grids include source factor defaults where defined,
reciprocal balance, singular-value exponent, whitening, damping, gamma, variance
coefficient, source subspace variants and preprocessing budget. IniLoRA includes
the complete 20,000-step Adam/StepLR source prefit, plus0/1000/4000; grouped weight
std is rounded to3decimals and the preprocessing cost is reported.

Centered lane uses V=W+(BA-B0A0), with both products evaluated under identical
ambient autocast. Original W and original magnitude are retained. This is a
representation adaptation of residual-base methods, not bitwise-equivalent BF16
residual storage. It adds a second factor product to every forward, which is
included in training time. Even when a factor is zero, common data callbacks
retain this documented runtime representation.

Warm-start lane retains intentional nonzero products for LoRA-One, LoRA-DA and
CNTLoRA cross/self/shift. It uses V=W+BA and original-W magnitude. It deliberately
does not require initial-function parity; initial output perturbation and quality
are recorded. Accepted column-normalized DoRA remains a geometry adaptation of
source LoRA, and differs from LoRA-One's official row-normalized DoRA branch.
CNT's unresolved source sign discrepancy is recorded; registered sign is
target_minus_base, with no post-result sign switching.

Quantization lane uses dense NF4 simulation, 4bits, flat blocks64, no double
quantization, iterations1/3, reciprocal balances.25/.5/1/2, original or quantized
magnitude. Each treatment is compared to a same-LR quantizer-only control sharing
its final dense base (iteration3 uses paired_loftq); original BF16 accepted DoRA
is also retained. Approximate reconstruction is measured, not asserted identical.
Dense storage gives no compressed-memory or serving-speed claim.

Scope exclusions are explicit: full EVA rank redistribution, TLoRA factor
freezing/rank allocation, MiCA freezing, LoRA-SB's third matrix and LoRA-Pro's
optimizer are structural/optimization changes, not this initializer study.
Gap-Init/GME-Init/extreme-subspace leads lack verified applicable recipes in the
finite primary-source inventory and are listed as untested, never as completed.

## Task and calibration protocol

| Task | Train / validation / test | Fit budget | LR grid | Fixed LR |
|---|---|---|---|---|
| SciFact rank8 | 726 / 81 / 300 queries | 5epochs,115updates | .0001,.001 | .001 |
| NFCorpus rank8 | 2582 / 323 / 323 queries | 5epochs,405updates | .00003,.0001,.0003,.001 | .0001 |
| SciFact transfer | NFCorpus fit / no tuning / 300queries | no extra fit | NFCorpus selection | .0001 |
| Aircraft rank2 | 3334 / 3333 / 3333images | 10epoch pilots,20epoch finals | .00001,.0001,.0003,.001 | .001 |
| ViGGO rank8 | 1024 / 128 / 256examples | 48update pilots,256update finals | .0001,.0003 | .0003 |

Use the previously pinned MiniLM, ViT and Qwen model/tokenizer/data revisions,
deduplicated splits, training loops, optimizers, schedules, checkpoints and task
metrics. SciFact uses official train/test with training-derived validation;
NFCorpus/Aircraft retain recorded official splits; ViGGO uses recorded subsets
of official splits. Full binding comes from task manifests and source snapshots.
Pilot seed42; final seeds42,43,44. Retrieval pilots use full fit budgets, so a
selected seed42 checkpoint is reused for final scoring only after selections
freeze. Aircraft/ViGGO finals retrain from scratch at full budget. Fixed/tuned
lanes may share an identical selected fit; report unique fit counts accurately.

After the main centered grid, a separate confirmation uses five additional
training seeds45–49. For each of the four training tasks choose the single best
centered initializer+LR using pilot validation score only (same tie-breaks as the
main selection), and compare it with accepted_anchor and accepted_gaussian_scale
at their validation-selected rates/settings. Deduplicate identical choices.
Transfer inherits the NFCorpus choices. Reuse full final training budgets and
task splits, collect seed-specific Aircraft calibration when needed, and retain
paired predictions. This tests sensitivity to the tuning seed; it is not a new
test dataset. Freeze a separate confirmation registry from validation data before
its training, with no test-score choice or post hoc replacement of the finalist.

Calibration takes128 training examples, deterministic seed20261009, batches8,
evaluation mode and the actual task loss. No validation/test examples, labels or
activations enter initialization. Collect unscaled task gradients, pooled masked
moments and separately source-specific unmasked batch Grams. Retain IDs, label
and attention masks/counts, losses and reduction weights, per-module checksums
and base/RNG preservation. Aircraft calibrates independently for each final seed
because its random classifier differs. Source-specific CorDA/Astra normalization,
TLoRA centered batch Grams, FILet input/output-gradient moments and raw CNT Grams
are retained. LoRA-DA N is full training population, never the128example subset;
common eval mode is an explicit difference from its source train-mode collector.

Cache only source/spec/model/calibration/seed-bound results. Prefit/decomposition
reuse must preserve factors and report cache hits, source preprocessing time and
actual wall time separately. IniLoRA prefit caching excludes downstream gain and
balance but binds all optimization inputs, device, TF32 policy and source.

## Gates, failure handling and reporting

Before main fits require CPU/CUDA independent formula and gradient checks,
source-specific calibration reference tests, deterministic RNG preservation,
accepted task trajectory parity, per-recipe actual model initial probes and
checkpoint recreation. Centered recipes require exact accepted initial function;
warm/quantized recipes require measured finite drift and deterministic expected
frozen weights. No trainable parameter count/order changes are allowed.
Frozen weights must remain unchanged during every fit. Quantized expected base
is separately bound to the quantizer recipe and checked on recreation.

Retain numerical initialization/training failures as registered failed cells,
with step count and reason; do not retry with altered epsilon, damping or seeds.
Ordinary implementation/infrastructure errors stop the shard and must be fixed
with new source-bound evidence. A wholly failed family has unavailable results.
Selection uses validation nDCG@10 (retrieval), macro accuracy then CE (Aircraft),
and validation NLL (ViGGO), deterministic lowerLR then candidateID tie-breaks.

Retain exact commands, hardware/runtime, source hashes and execution snapshot,
calibration tensors, initial factor hashes/statistics, training logs, trainable
checkpoints, reload probes and per-example predictions. Independent auditors
must rescore all final predictions and check archived provenance/selection.
Report all families, fixed and tuned lanes, candidate counts, selected parameters,
quality mean/sampleSD, failure counts and preprocessing/training/evaluation costs.

Primary contrasts compare against exact accepted anchor and equally searched
original-draw Gaussian. Quantized contrasts additionally use paired quantizer
controls; centered versus warmstart shares the same factor recipe/settings.
Previously inspected test sets and many searched families make this exploratory.
Intervals are paired seed/example bootstrap,10000draws, Aircraft stratified by
class; they are descriptive and unadjusted, never blanket superiority evidence.
Include all family results in a machine-readable table; README/main chart may
show declared representative source settings and selected family summaries with
links to the full table. Do not select a new public default or push changes.
