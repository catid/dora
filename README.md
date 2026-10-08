# DoRA

A small PyTorch implementation of [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) (Liu et al., 2024).

For a PyTorch linear weight `W` of shape `[out_features, in_features]`, the adapted weight is

```text
V = W + A @ B
W_adapted = m * V / stop_gradient(row_norm(V))
y = linear(x, W_adapted, bias)
```

There is one trainable magnitude per **output**, with `m.shape == (out_features, 1)`. The denominator is detached as in the paper's memory-efficient variant (§4.3), also used by [Hugging Face PEFT](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py). The base weight and bias are frozen. This implementation retains its original factor naming: `A` has shape `[out_features, rank]`, `B` has shape `[rank, in_features]`, and the update has implicit scaling 1.

## Usage

```bash
python -m pip install -r requirements.txt
python dora.py
python -m unittest -v test_dora
```

```python
import torch
from torch import nn
from dora import DoRALayer, replace_linear_with_dora

base = nn.Linear(128, 64)
layer = DoRALayer.from_linear(base, rank=8)
# The initial adapter preserves base(x), including its bias.
x = torch.randn(16, 128)
torch.testing.assert_close(layer(x), base(x))

model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16))
model = replace_linear_with_dora(model, rank=8)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=1e-3
)

# After fine-tuning, export an independent frozen Linear for inference.
merged = layer.eval().to_linear()
torch.testing.assert_close(merged(x), layer(x))
```

Always use the return value of `replace_linear_with_dora`: the root itself may be a linear layer. Nested modules and shared module aliases are supported. Conversion copies the source tensors and preserves device, weight dtype, bias presence, and training/evaluation mode. Parameters in other kinds of layers keep their existing `requires_grad` settings; freeze those separately if needed. Distinct linear modules with tied weight parameters receive independent copies. Recreate the optimizer after conversion.

`DoRALayer(d_in, d_out, rank=4)` also works directly: it initializes a frozen base weight like `nn.Linear` and has no bias unless a bias tensor is supplied. This is primarily an adapter for pretrained weights.

For FP16/BF16, convert the base layer to its intended dtype **before** wrapping it. Norms and initial magnitude parameters use FP32, and magnitude rescaling occurs before casting activations back. Calling `.half()` on the adapter afterward also casts its magnitude parameter under normal PyTorch behavior. Zero norms and initial magnitudes use the same small positive floor, preserving initial outputs while allowing zero rows to learn. Extreme values can still overflow or underflow during ordinary floating-point norm reductions; this implementation targets ordinary pretrained weight ranges. Merged and factorized outputs can differ slightly due to floating-point evaluation order.

## Corrections and validation

The original code normalized `dim=0` even though PyTorch stores transposed linear weights. Its one-output demo consequently reduced direction adaptation to elementwise signs, with nearly zero direction gradients. This version normalizes `dim=1`, detaches the denominator, initializes standalone weights, handles missing biases, and avoids resetting the caller's RNG on import. **Old adapter checkpoints are incompatible with the corrected magnitude layout.**

The 20 tests cover an independent dense forward/gradient reference with nonzero adapters, conversion identity, frozen base weights, optimizer updates, root/shared module conversion, state dictionaries, merged inference, zero/tiny weights, CPU/CUDA execution, autocast, and FP16 overflow regressions. They passed locally with PyTorch 2.12.0+cu130, including FP32, FP64, FP16 and BF16 checks.

The demo evaluates the same held-out dataset before and after adaptation and checks conversion identity. A seeded local run produced:

```text
Total Parameters: 11
Trainable Parameters: 11
Final Evaluation Loss: 0.15444783866405487
Total Parameters: 56
Trainable Parameters: 45
Continuing training with DoRA layers...
Final (DoRA) Evaluation Loss: 0.08973299711942673
```

This synthetic example is a smoke test, not a model-quality benchmark. Exact losses can vary by PyTorch version and hardware.

## Performance

Training applies the low-rank update to activations, keeping the dense adapted weight out of the backward graph. The row norm still requires a dense temporary, computed without autograd and reused for the base-weight addition. `to_linear()` folds the adapter into a plain frozen linear layer for inference, eliminating ongoing adapter overhead; the export is a snapshot and does not track later training updates.

The benchmark compares against a **corrected dense DoRA reference**, validates outputs and input/adapter gradients first, and saves raw timings, peak incremental CUDA allocations, source hashes, seed, commands, and environment details:

```bash
python benchmark_dora.py --device cuda:0 --output benchmark_1024.json
python benchmark_dora.py --device cuda:0 --in-features 4096 --out-features 4096 \
    --tokens 32 --output benchmark_4096.json
```

Timing includes synchronized forward/backward calls but excludes optimizer steps. Merged inference excludes one-time export cost. Performance depends on matrix size and batch size; extra kernel launches can make factorization slower for small layers.

Measured on an NVIDIA RTX PRO 6000 Blackwell Max-Q with PyTorch 2.12.0+cu130, FP32, rank 8 (median of seven samples, 20 calls per sample):

| Weight / tokens | Dense forward + backward | Factorized forward + backward | Dense / factorized peak allocation |
| --- | ---: | ---: | ---: |
| 1024 × 1024 / 32 | 0.203 ms | 0.250 ms | 16.1 / 4.0 MiB |
| 1024 × 1024 / 256 | 0.204 ms | 0.259 ms | 17.0 / 6.0 MiB |
| 4096 × 4096 / 32 | 0.652 ms | 0.256 ms | 256.5 / 64.0 MiB |

The largest case is **2.55× faster with 75% less incremental peak allocation**. Its merged inference takes 0.061 ms versus 0.231 ms for the dense adapter. These allocation figures exclude existing model/input tensors and optimizer state. Small cases favor dense computation for latency. Worst FP32 output/gradient relative-L2 error was 6.2e-7; a separate BF16 smoke check also passed. Full configurations, comparisons, raw timing samples, and provenance are in [benchmark_results.json](benchmark_results.json).

## Real-world task comparisons (2026-10-08)

**NoRA produced the clearest gains on retrieval and JSON extraction.** DoRA had the highest Flowers102 accuracy, with small differences near the task's accuracy ceiling. Combining DoRA and NoRA did not consistently improve on NoRA.

We ran four tasks on this machine's RTX PRO 6000 Blackwell Max-Q GPUs, with **three training seeds per method**, rank 8, matched initializations, and equal validation-only learning-rate searches within each task. These are local, limited-budget comparisons of LoRA, DoRA, [NoRA](https://arxiv.org/abs/2608.31036), and DoRA+NoRA; they do not reproduce the NoRA paper's benchmark scores. Models, data, precision, and commands are in the [experiment protocol](experiments/README.md).

| Method | Flowers102 top-1 % ↑ | SciFact nDCG@10 ×100 ↑ | ViGGO exact match % ↑ | SDXL denoising MSE ↓ |
|:--|--:|--:|--:|--:|
| Task baseline | 98.88 ± 0.12 | 64.63 | 1.56 | 0.1047 |
| LoRA | 99.50 ± 0.02 | 65.10 ± 0.31 | 64.58 ± 4.32 | 0.1023 ± 0.0001 |
| DoRA | 99.58 ± 0.02 | 65.38 ± 0.32 | 64.84 ± 3.77 | 0.1023 ± 0.0001 |
| NoRA | 99.52 ± 0.15 | 68.47 ± 0.62 | 75.13 ± 3.63 | 0.1051 ± 0.0013 |
| DoRA+NoRA | 99.45 ± 0.24 | 68.63 ± 0.48 | 69.01 ± 6.79 | 0.1053 ± 0.0003 |

Adapter values are mean ± sample standard deviation across seeds 42–44. The vision baseline trains only the classifier with the same three seeds; the other baselines are single evaluations of frozen pretrained models. SciFact nDCG is multiplied by 100. SDXL MSE is a denoising objective, **not an image-quality score**.

![Four-task comparison with training-seed error bars](results/2026-10-08/comparison.png)

- **Retrieval:** NoRA improved nDCG@10 by **3.37 points** over LoRA; the exploratory paired-bootstrap 95% interval was **[1.43, 5.32]**. Adding DoRA gave only **0.16 points**, with an interval spanning zero.
- **JSON extraction:** NoRA improved exact match by **10.55 percentage points** over LoRA; the corresponding interval was **[3.13, 18.75]**. DoRA alone showed no clear advantage over LoRA.
- **Vision:** DoRA led LoRA by **0.076 percentage points**. All four adapter methods exceeded 99.4% mean accuracy, leaving little room to distinguish them.
- **SDXL:** LoRA and DoRA were nearly tied on held-out denoising loss. This study used one subject and just **3 train / 1 validation / 1 test photos**. See the [fixed-prompt sample grid](results/2026-10-08/sdxl_samples.jpg) and [separate image diagnostics](results/2026-10-08/sdxl_metrics.md); neither loss nor similarity establishes an overall image-quality winner.

Measured median training time, in seconds:

| Method | Flowers102 train+val (s) | SciFact train (s) | ViGGO train+val (s) | SDXL train (s) |
|:--|--:|--:|--:|--:|
| Task baseline | 29.2 | — | — | — |
| LoRA | 55.0 | 1.6 | 59.2 | 135.8 |
| DoRA | 68.1 | 1.8 | 61.0 | 184.6 |
| NoRA | 55.6 | 1.7 | 59.7 | 180.6 |
| DoRA+NoRA | 68.9 | 1.9 | 61.5 | 229.4 |

Timing excludes downloads, model loading, and test-image/text generation; vision and extraction include validation. DoRA's magnitude scaling and dense weight-norm calculation add training work. Exact timing scopes, trainable-parameter counts, and peak CUDA allocations are retained in the [machine-readable results](results/2026-10-08/summary.json).

The combined core/adapter suite passed **26 tests**. An [independent artifact audit](results/2026-10-08/validation.json) recomputed every reported primary score from saved predictions or noise probes and checked recorded source hashes. [Raw artifacts](results/2026-10-08/raw_artifacts.tar.gz) include predictions, logs, split manifests, and executed source snapshots; [their manifest](results/2026-10-08/raw_manifest.json) records hashes. The [archive-only reproduction command](results/2026-10-08/reproduce.md) rebuilds the tables and chart without downloading models. Checkpoints and full-resolution image sets remain in the local run directories. Three seeds, small evaluation sets, and a two-rate search limit the conclusions; no method is a universal winner.
