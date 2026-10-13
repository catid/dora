# Next change after FP32 magnitude storage

The baseline is the current `dora.py`: original column normalization, full norm
derivative, historical CPU normal-up/zero-down initialization and dense forward,
with native initial norm computation followed by FP32 magnitude storage (FP64
retained). The experiment adapter shares the existing pretrained-model device,
biasless-layer and projection-dtype compatibility handling across every recipe.
The public implementation is not changed by this study.

Prespecified candidates each add exactly one change to that baseline:

| Method | Additional change |
| --- | --- |
| `next_base` | None |
| `next_axis` | Output-row rather than input-column normalization |
| `next_detach` | Stop gradients through the denominator |
| `next_norm_fp32` | FP32 initial and denominator norm work; retain FP64 |
| `next_init` | Existing task-harness Kaiming down/zero-up initializer |
| `next_factorized` | Factorized projections with full norm gradients retained |

The initializer treatment includes the historical harness's orientation,
distribution, RNG device and parameter-registration policy. Factorization also
changes floating-point association and projection precision. Neither is claimed
to isolate a smaller subcomponent. No norm floor is enabled in these recipes.
The axis change also changes the number of magnitudes in nonsquare layers;
equal rank does not imply equal trainable parameter count.

Use the previously pinned models, dataset revisions and exact split IDs. Seeds
42, 43 and 44 are paired. All learning-rate and checkpoint selection uses only
validation data. Test sets were inspected in earlier studies, so this is an
exploratory comparison, not a fresh confirmatory benchmark.

| Task | Train / validation / test | Budget | LR selection |
| --- | --- | --- | --- |
| SciFact r8 | 726 / 81 / 300 queries | 5 epochs, 115 updates | Seed42 full runs at 0.0001 and 0.001 |
| NFCorpus r8 | 2582 / 323 / 323 queries | 5 epochs, 405 updates | Seed42 full runs at 0.00003, 0.0001, 0.0003 and 0.001 |
| SciFact transfer r8 | NFCorpus training / no extra tuning / 300 | Same NFCorpus fit | NFCorpus selection |
| Aircraft r2 | 3334 / 3333 / 3333 images | 20 epochs | Four seed42 ten-epoch pilots at 0.00001, 0.0001, 0.0003 and 0.001 |
| ViGGO r8 | 1024 / 128 / 256 examples | 256 updates | Two seed42 48-step pilots at 0.0001 and 0.0003 |

SciFact uses official train/test with a recorded validation subset of training;
NFCorpus and Aircraft use their recorded deduplicated official splits. ViGGO uses
recorded deterministic subsets of its official splits. Preserve both fixed
baseline LR (0.001 SciFact/Aircraft, 0.0001 NFCorpus, 0.0003 ViGGO) and equally
validation-tuned comparisons. Identical fits shared between the two comparisons
count once. SciFact transfer adds no fit.

Retrieval and ViGGO are freshly trained, including the baseline. Aircraft's FP32
weights make magnitude storage promotion an exact no-op for each candidate.
Reuse its old fits only after proving bitwise initialization, RNG, outputs,
gradients and clipped AdamW updates on three actual Aircraft minibatches, then
re-auditing all selected source fits and predictions. Reused results retain
their original method names, fit IDs and source hashes in the evidence.

Report per-task mean, sample SD and candidate-minus-current-baseline effects.
Exploratory paired 95% intervals resample seeds and examples (stratified by class
for Aircraft), with 10,000 draws and no multiple-comparison adjustment. Do not
combine the task scores into an arbitrary overall score. Recommend an additional
default change only if the observed gains and regressions support it; an
inconclusive result is valid. Keep task-quality timing separate from claims about
production throughput, since these runs include shared norm instrumentation.

A separate layer microbenchmark measures all six recipes on GPU3 with the
read-only norm instrumentation disabled after bitwise output/all-gradient checks.
It covers 1024-square weights with 32 or 256 tokens and 4096-square weights with
32 tokens, rank8, FP32 or BF16 frozen weights/inputs under BF16 autocast (FP32
factors/magnitudes). Each case uses ten warmups and seven samples of twenty
synchronized calls, with input gradients and no optimizer steps. Report raw
samples and incremental allocated memory above existing tensors. This measures
shape-dependent efficiency, separately from downstream quality.
