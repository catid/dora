# Normalize the up factor while retaining accepted DoRA initialization

The selected public implementation stays original `dora.py` plus FP32 magnitude
storage (FP64 retained). This experiment adds only differentiable rank-dimension
normalization of the existing nonzero up factor; it keeps zero down, normal up,
input-column DoRA weight normalization, full denominator gradients, native norm
arithmetic, dense projection, implicit scale 1, and the original RNG/order policy.

With conventional names `up=[out,rank]`, `down=[rank,in]`, the treatment is:

```python
delta = F.normalize(up, p=2, dim=1, eps=1e-12) @ down
V = W + delta
adapted = m * (V / V.norm(p=2, dim=0, keepdim=True))
```

The public class calls these factors `lora_A` (up) and `lora_B` (down). The task
adapter names are reversed. Normalizing up rows across rank equals normalizing
columns of `up.T` in the transposed factorization. This is an up-factor,
NoRA-inspired variant, distinct from the paper's down-column normalization.
Both recipes start with a zero adapter update. The normalized factor begins
nonzero, so this treatment avoids the previously observed zero-down epsilon
start. It still changes effective scaling and optimization geometry; quality
improvement is a hypothesis to measure.

Fresh recipes: `up_base` (accepted DoRA) and `up_nora` (up normalization added).
Compare both at the fixed baseline LR and after equal validation-only tuning.
Keep the preceding model/data revisions, splits, ranks, budgets and seeds 42–44:

| Task | Train / validation / test | Budget | LR grid | Fixed LR |
| --- | --- | --- | --- | --- |
| SciFact r8 | 726 / 81 / 300 queries | 5 epochs, 115 updates | 1e-4, 1e-3 | .001 |
| NFCorpus r8 | 2582 / 323 / 323 queries | 5 epochs, 405 updates | 3e-5, 1e-4, 3e-4, 1e-3 | .0001 |
| SciFact transfer r8 | NFCorpus fit / no tuning / 300 | No extra fit | NFCorpus selection | .0001 |
| Aircraft r2 | 3334 / 3333 / 3333 images | 20 epochs; 10-epoch pilots | 1e-5, 1e-4, 3e-4, 1e-3 | .001 |
| ViGGO r8 | 1024 / 128 / 256 examples | 256 updates; 48-step pilots | 1e-4, 3e-4 | .0003 |

SciFact uses official train/test with training-derived validation. NFCorpus and
Aircraft retain recorded deduplicated official splits. ViGGO uses the recorded
deterministic subsets of official splits. No test score selects an LR/checkpoint.
For retrieval and Aircraft each method finishes its pilot grid before test
evaluation; ViGGO completes all four pilots before test generation.

Require CPU/CUDA formula and gradient checks, exact task-level baseline controls,
finite training, audited checkpoints and independently rescored predictions.
Verify the fresh baseline against preceding `current_dora` full-fit tensors and
predictions. Reuse earlier down-normalized and alternate-initializer results only
as explicitly historical comparisons with bound evidence; never count them as
new training replicates. The primary contrast is `up_nora - up_base`.

These are exploratory evaluations on previously inspected test sets. Report
means and sample SD over seeds 42–44, plus 10,000 crossed paired seed/example
bootstrap draws (class-stratified Aircraft), with no multiplicity correction.
Keep all earlier sources/results frozen. Retain commands, source hashes, logs,
checkpoints and predictions, and independently replay the report archive.
