"""Historical DoRA adapter for comparisons with commit bb97617.

The dense forward below is transcribed without mathematical changes: A has
shape [out, rank], B has shape [rank, in], magnitudes index input columns,
and the column norm retains its gradient. There is no normalization epsilon,
precision promotion, detached denominator, or autocast override.

Compatibility changes are confined to accepting a pretrained Linear, preserving
its frozen weight/bias storage and dtype, supporting absent bias, and moving
CPU-initialized factors to the weight device. Magnitudes retain the base dtype;
factors retain PyTorch's default floating dtype (FP32 in the task protocols).
The unmodified historical F.linear cannot accept BF16 activations with FP32
effective weights outside autocast. For that otherwise invalid call only, the
projection casts input/bias to the effective weight dtype and casts its result
back to the input dtype. This adds output rounding at a documented portability
boundary; it does not cast the factors or adapted weights, promote the norm, or
change its gradient. Matching dtypes and existing autocast retain the exact
historical projection behavior. Importing this module does not reset any RNG.

The existing task adapters use different factor orientation/initialization and
FP32 magnitudes. Those choices are intentionally not inherited here. The frozen
historical source is archived at ORIGINAL_SOURCE_PATH, and tests compare its
actual class initialization, forward, and gradients against this implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F


ORIGINAL_COMMIT = "bb97617a0d5e1ad4c6856cb7278f5a7386820d18"
ORIGINAL_SOURCE_SHA256 = "e1d25979459dce1ff5f458469bece1e928c8bae02844acc6259b606ac228174d"
ORIGINAL_SOURCE_PATH = "results/2026-10-09-baseline-audit/original_dora.py"
COMPATIBILITY_CHANGES = (
    "Accept an existing pretrained nn.Linear instead of uninitialized standalone weights.",
    "Retain the supplied frozen weight/bias storage and dtype; absent bias stays None.",
    "Draw original-shaped factors on CPU using the default floating dtype, then move only their device.",
    "Do not reset the caller's RNG when importing the module.",
    "Outside autocast only, cast mismatched projection input/bias to the effective weight dtype and return the input dtype.",
    "Expose benchmark metadata and a dense merge helper without changing the training forward.",
)


def _project(x, weight, bias):
    """Accommodate only the original dense projection's dtype boundary."""
    mismatch = x.dtype != weight.dtype or (bias is not None and bias.dtype != weight.dtype)
    if mismatch and not torch.is_autocast_enabled(x.device.type):
        projected_bias = None if bias is None else bias.to(weight.dtype)
        return F.linear(x.to(weight.dtype), weight, projected_bias).to(x.dtype)
    return F.linear(x, weight, bias)


class _MergedOriginalLinear(nn.Linear):
    def forward(self, x):
        return _project(x, self.weight, self.bias)


class OriginalDoRALinear(nn.Module):
    """Original column-normalized, full-norm-gradient DoRA for a frozen Linear."""

    method = "dora_original"
    use_original = True
    use_dora = True
    use_nora = False

    def __init__(self, base, rank=8):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not base.weight.is_floating_point():
            raise ValueError("base weight must have a floating-point dtype")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.weight = base.weight
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Keep the original norm operation, reduction axis, and storage dtype.
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        # CPU draws reproduce the historical constructor's RNG sequence, even
        # when the frozen model has already been moved to an accelerator.
        std_dev = 1 / torch.sqrt(torch.tensor(rank, device="cpu").float())
        a = torch.randn(self.out_features, rank, device="cpu") * std_dev
        b = torch.zeros(rank, self.in_features, device="cpu")
        self.lora_A = nn.Parameter(a.to(device=self.weight.device))
        self.lora_B = nn.Parameter(b.to(device=self.weight.device))
        self.register_parameter("log_gain", None)
        self.train(base.training)

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return _project(x, calc_weights, self.bias)

    @classmethod
    def from_linear(cls, layer, rank=8):
        return cls(layer, rank=rank)

    @torch.no_grad()
    def to_linear(self):
        """Merge the same dense formula in the caller's ambient precision scope.

        No dtype casts are introduced: the effective weight may have a higher
        precision than the frozen base, as it does in the original forward.
        The projection retains the same explicit mixed-dtype boundary handling.
        """
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        merged = _MergedOriginalLinear(self.in_features, self.out_features,
                                       bias=self.bias is not None, device="meta",
                                       dtype=calc_weights.dtype)
        merged.weight = nn.Parameter(calc_weights, requires_grad=False)
        if self.bias is not None:
            merged.bias = nn.Parameter(self.bias.detach().clone(), requires_grad=False)
        return merged.train(self.training)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, bias={self.bias is not None}, method={self.method!r}")
