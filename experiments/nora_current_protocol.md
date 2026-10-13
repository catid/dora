# NoRA added to the accepted DoRA implementation

The accepted DoRA is the original `dora.py` plus FP32 magnitude storage (FP64
retained). Keep its input-column weight normalization, full denominator gradient,
dense forward and original initialization. The study does not change that file.

NoRA normalizes each column of the input/down factor across its rank dimension,
with the full normalization derivative. This is `lora_A` in the task adapters and
`lora_B` in the original public class. Use `F.normalize(..., p=2, dim=0,
eps=1e-12)` consistently with the prior NoRA experiments. It is used inside the
current DoRA adapted-weight construction and full-gradient weight norm.

The original initializer has a zero input/down factor. Thus a literal NoRA
addition starts inside the epsilon branch: its zero-factor derivative is scaled
by `1/eps`, followed by the unchanged global gradient clipping and AdamW. This is
a deliberate, finite, epsilon-defined treatment with the original initializer;
it is not the paper's random-input/zero-output initialization. Do not hide the
initialization change when evaluating a conventional NoRA combination.

Run the full two-by-two design:

| Method | Initialization | NoRA input normalization |
| --- | --- | --- |
| `current_dora` | Original normal up, zero down | Off |
| `current_dora_nora` | Original normal up, zero down | On |
| `init_dora` | Existing Kaiming down, zero up recipe | Off |
| `init_dora_nora` | Same Kaiming down, zero up recipe | On |

All methods retain FP32 magnitudes, input-column DoRA normalization, native
weight-norm arithmetic, full weight-norm gradients and dense projection. The
alternate initializer includes the prior harness's orientation, distribution,
RNG device and parameter-registration policy. Shared pretrained-model handling
preserves device/storage, absent bias and projection dtype compatibility.

Preserve all models, dataset revisions, split IDs, ranks, training budgets,
optimizer groups and seeds 42–44 from the previous next-change evaluation:

| Task | Train / validation / test | Budget | Equally budgeted validation LR search |
| --- | --- | --- | --- |
| SciFact r8 | 726 / 81 / 300 queries | 5 epochs, 115 updates | Seed42 full fits: 0.0001, 0.001 |
| NFCorpus r8 | 2582 / 323 / 323 queries | 5 epochs, 405 updates | Seed42 full fits: 0.00003, 0.0001, 0.0003, 0.001 |
| SciFact transfer r8 | NFCorpus training / no tuning / 300 | Same NFCorpus fits | NFCorpus selection |
| Aircraft r2 | 3334 / 3333 / 3333 images | 20 epochs | Seed42 ten-epoch pilots: 0.00001, 0.0001, 0.0003, 0.001 |
| ViGGO r8 | 1024 / 128 / 256 examples | 256 updates | Seed42 48-step pilots: 0.0001, 0.0003 |

SciFact uses official train/test with validation carved from training; NFCorpus
and Aircraft use the recorded deduplicated official splits. ViGGO uses recorded
deterministic subsets of its official splits. This is a repeat of local task
protocols, not a replication of the NoRA paper's reported benchmark scores.

Train all four recipes freshly. Verify NoRA-off baseline and initializer controls
against the previously measured recipes, using full-model step controls and
complete checkpoint/prediction reproduction where applicable. Retain both the
fixed current-baseline LR (0.001 SciFact/Aircraft, 0.0001 NFCorpus, 0.0003 ViGGO)
and validation-tuned comparisons. Identical fits serving both lanes count once.
Test data never chooses a rate or checkpoint. Log initial zero-factor/epsilon
behavior explicitly; zero-factor NoRA epsilon activation is not a DoRA norm-floor
failure. Reject nonfinite losses, gradients, parameters or predictions.

For retrieval and Aircraft, each recipe's complete pilot grid precedes that
recipe's final test evaluation; recipes run in a predeclared order without
adapting subsequent searches to earlier test scores. ViGGO completes all eight
pilots across all recipes before any final test generation.

Primary contrast: `current_dora_nora - current_dora`. Initialization-controlled
contrast: `init_dora_nora - init_dora`. Also report `init_dora - current_dora`
and the complete combined recipe versus the accepted default, clearly labeled.
Report per-task means, sample SD and paired seed/example bootstrap intervals
(10,000 draws, stratified by Aircraft class), with three seeds and no
multiple-comparison correction. These previously inspected tasks make the study
exploratory. Keep raw source snapshots, commands, logs, checkpoints and
predictions; independently rescore and replay the final tables/charts.
