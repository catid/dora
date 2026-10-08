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
