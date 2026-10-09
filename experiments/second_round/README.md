# Harder tasks and experimental DoRA/NoRA mixtures

This round preserves the original experiments. It separates capacity, optimization, and generalization effects using tasks with more room to improve. Neither new mixture is assumed to win.

## Methods and controls

The shared [adapters](../second_round_adapters.py) use Kaiming input factors, zero output factors, effective scaling 1, no dropout, and matched base checkpoints. Two experimental variants join the original four methods:

| Name | Change from DoRA+NoRA | Extra parameters |
| --- | --- | --- |
| Slower magnitudes (`dora_nora_mlr`) | Magnitude learning rate is 0.01× or 0.1× the factor rate, selected on validation data. Forward behavior is identical. | None |
| Input gains (`dora_nora_gain`) | Each normalized input-factor column has a positive learned gain, initialized to 1. | One per adapted input feature |

For conventional factors `A=[rank,input]`, `B=[output,rank]`, the gain variant is:

```text
A_eff[:,j] = A[:,j] / max(norm(A[:,j]), eps) * exp(log_gain[j])
V = W + B @ A_eff
W_adapted = m * V / stop_gradient(row_norm(V))
```

NoRA normalization remains differentiable. Magnitudes and gains have zero weight decay in **every second-round method**; factor decay is shared within a task. This is an explicit optimizer difference from round 1. Task heads have a separate fixed learning rate. See [optimizer integration](optimizer.md) for PyTorch 2.12's named-group requirement.

Gains restore column-amplitude freedom while adding parameters and changing optimization. Equal rank is not equal parameter count. Every method receives four validation configurations per task/rank: ordinary/gain methods try four base rates; slower magnitudes try two base rates crossed with two magnitude multipliers. These are equally budgeted limited searches, not exhaustive searches or equal-quality runtime measurements. Test data never select rates or checkpoints.

The gain model has the same attainable weight family as ordinary DoRA at fixed rank; it uses a different amplitude/direction parameterization. Its gains start at 1, so its initial effective input factors match NoRA rather than ordinary DoRA. Matching raw random factors and mathematical initial outputs does not match the first output-factor gradient scale. See the [mathematical review](mixing_notes.md) for the full-rank constraint, unused-rank construction, and limits of the capacity interpretation.

## Fixed protocols

| Task | Model and data | Training and selection | Measurements |
| --- | --- | --- | --- |
| Aircraft variants | ViT-B/16; official 3,334 train / 3,333 validation / 3,333 test images, 100 classes; remove copyright band | Ranks 2/8; 20 epochs, batch 128; four 10-epoch trials; select validation macro accuracy then cross-entropy | Top-1 and macro accuracy |
| Biomedical retrieval | MiniLM-L6-v2; NFCorpus 3,633 documents, 2,582 train / 323 validation / 323 test queries after exact-text exclusions | Ranks 2/8; 5 epochs, batch 32; four fixed frozen-model hard negatives; four full-horizon trials selected by validation nDCG | NFCorpus nDCG@10 and untouched SciFact transfer |
| Compositional parsing | Qwen2.5-3B-Instruct; COGS 4,096 train / 128 IID validation / 128 IID test / 32 per generalization category | Rank 8; 384 updates, batch 16; four full-horizon trials selected by target-token validation NLL | Atom-set exact match; strict-string and lexical/structural scores separately |
| Controlled teachers | 28 fixed cells crossing ranks 2/8, row scaling, directional changes, column amplitudes, and coordinates | 800 full-batch updates; four full-horizon validation trials | Test MSE / frozen-model MSE, stratified by family and rank |

All final comparisons use seeds 42/43/44. Retrieval, COGS, and teachers reuse the winning seed-42 checkpoint because tuning and final budgets match. Aircraft refits all three seeds; its 10-epoch selection horizon can choose a suboptimal rate for 20 epochs. Aircraft rank shards independently repeat the same three trained-head baselines, identified separately rather than counted as six independent seeds.

COGS retains all 155 rare primitive/exposure examples and uses three fixed demonstrations from its training set. Generation is greedy, capped at 640 new tokens. Every selected gold target fits; no input/target is silently truncated. Atom-set exact match ignores conjunct order and whitespace but preserves predicates, roles, argument order, variable indices, and definite markers. Invalid syntax is rejected. All 21 official generalization categories are included, with lexical and structural scores separate. This is a pretrained subset experiment, not a reproduction of the original COGS paper. Pretraining exposure cannot be ruled out.

The captured COGS runner's `generation_cap_hits` diagnostic checks absence of tokenizer EOS 151645, although the pinned model also stops on EOS 151643. Its recorded flags can therefore include false positives and must not be treated as exact counts of length exhaustion. Raw generated token IDs were not saved, preventing an exact retrospective correction. Text scores and validation selection do not depend on this diagnostic; their saved predictions remain fully auditable.

The current runner fixes this diagnostic for future runs and retains raw continuation IDs and the complete configured EOS list. This change was made after all 18 final COGS runs; the archive preserves their executed source. See the [diagnostic audit](generation_diagnostics.md).

Aircraft initializes adapters on CPU before moving them to GPU. A separate zero-update control on all 3,333 validation images found no FP32 prediction changes relative to the native model, but BF16 LoRA/NoRA and the four DoRA-based methods form two slightly different numerical families: 43 predictions differ between them. The trained-head native model gets 1,451 correct, LoRA/NoRA 1,446, and the DoRA family 1,443. CPU-initialized magnitude divided by GPU-recomputed norm differs from one by at most 2.384e-7. On a separate 128-image mechanism probe, recomputing only a fresh DoRA model's magnitudes on GPU makes its logits bitwise equal to LoRA in both FP32 and BF16. These controls do not measure the effect on subsequent training. All trained results retain the original protocol; future experiments should initialize fresh adapters on their final device and dtype. Never reset learned magnitudes when moving or loading a trained adapter. Full arrays and [independent numerical audits](../../results/2026-10-08-round2/numerical_controls.md) are retained.

Teacher seeds vary initialization on one fixed generated problem per cell; seed SD does not estimate variation across a population of teacher tasks. White/rescaled pairs preserve the exact function and labels. Capacity statements concern NoRA's normalized branch outside its epsilon clamp, not an impossibility proof for the entire DoRA+NoRA model. The synthetic grid establishes no downstream winner.

## Reproduction

Install the [package pins](../../requirements-experiments.txt). Hardware and environment follow the [first-round protocol](../README.md). Use fresh output directories when changing configurations, and select only available GPUs. Every run records model/data revisions and source hashes.

```bash
export HF_HOME=/var/tmp/dora-bench/cache/huggingface
CUDA_VISIBLE_DEVICES=1 python -m experiments.second_round.retrieval
CUDA_VISIBLE_DEVICES=3 python -m experiments.second_round.teacher

python -m experiments.second_round.prepare_aircraft
CUDA_VISIBLE_DEVICES=0 python -m experiments.second_round.vision --phase run \
  --ranks 2 --output /var/tmp/dora-bench/round2/vision/rank2_run
CUDA_VISIBLE_DEVICES=1 python -m experiments.second_round.vision --phase run \
  --ranks 8 --output /var/tmp/dora-bench/round2/vision/rank8_run

# Compute the shared COGS baseline once, then run methods independently.
CUDA_VISIBLE_DEVICES=2 python -m experiments.second_round.cogs \
  --methods lora --output /var/tmp/dora-bench/round2/cogs/lora
CUDA_VISIBLE_DEVICES=3 python -m experiments.second_round.cogs \
  --methods nora --output /var/tmp/dora-bench/round2/cogs/nora \
  --baseline-from /var/tmp/dora-bench/round2/cogs/lora/baseline.json
```

Repeat the COGS command for `dora`, `dora_nora`, `dora_nora_mlr`, and `dora_nora_gain`, each with its own output directory and the shared baseline. All models use identical prompt-masked loss and demonstration/decoding settings.

Analyses reparse saved predictions, recompute metrics, verify source/checkpoint hashes and selection, and calculate paired bootstrap intervals. Intervals are exploratory and unadjusted for multiple comparisons; three training seeds provide limited evidence about seed variance.

```bash
python -m unittest discover -v
python -m experiments.second_round.analyze_cogs
python -m experiments.second_round.plot_teacher
```

After every task is complete and its audit passes, build the report and portable evidence bundle:

```bash
python -m experiments.second_round.report \
  --raw-root /var/tmp/dora-bench/round2 \
  --output-dir results/2026-10-08-round2
```

The [published reconstruction instructions](../../results/2026-10-08-round2/reproduce.md) rebuild scores and figures from saved predictions without model downloads. Downstream adapter checkpoints remain in the local run directories; the bundle retains their original local audit results and hashes. The compact teacher evidence includes all 504 small adapter checkpoints and a captured deterministic generator: its standalone CPU verifier regenerates every problem tensor, checks exact tensor hashes, and independently reconstructs the dense adapted weights and test errors. Library drift causes an explicit verification failure.
