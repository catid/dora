"""Weight-decomposed low-rank adaptation for PyTorch linear layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DoRALayer(nn.Module):
    """A frozen linear weight with trainable DoRA magnitude and direction.

    PyTorch stores weights as [out_features, in_features], so each *row* has
    one magnitude. The denominator is detached as in the memory-efficient
    variant in section 4.3 of the DoRA paper. The low-rank update is A @ B
    with implicit scaling 1; the historical factor names/shapes are retained.
    """

    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None, *, device=None, dtype=None):
        super().__init__()
        for name, value in (("d_in", d_in), ("d_out", d_out), ("rank", rank)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_features = d_in
        self.out_features = d_out
        self.rank = rank

        if weight is None:
            base_weight = torch.empty(d_out, d_in, device=device, dtype=dtype)
            if not base_weight.is_floating_point():
                raise ValueError("weight must have a floating-point dtype")
            nn.init.kaiming_uniform_(base_weight, a=math.sqrt(5))
        else:
            if weight.shape != (d_out, d_in):
                raise ValueError(f"weight must have shape {(d_out, d_in)}")
            if not weight.is_floating_point():
                raise ValueError("weight must have a floating-point dtype")
            base_weight = weight.detach().to(device=device, dtype=dtype).clone()
            if not base_weight.is_floating_point():
                raise ValueError("dtype must be floating-point")
        self.weight = nn.Parameter(base_weight, requires_grad=False)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            if bias.shape != (d_out,):
                raise ValueError(f"bias must have shape {(d_out,)}")
            if not bias.is_floating_point():
                raise ValueError("bias must have a floating-point dtype")
            self.bias = nn.Parameter(bias.detach().to(self.weight).clone(), requires_grad=False)

        # Initialize magnitudes before drawing adapters; clamp zero rows as well
        # so conversion is an identity and zero rows can receive adapter gradients.
        with torch.no_grad():
            # Retain FP32 magnitudes for half/bfloat16 weights: a row's norm
            # can exceed the weight dtype's range even when every entry fits.
            magnitude = self._row_norm(self.weight)
        self.m = nn.Parameter(magnitude)
        self.lora_A = nn.Parameter(self.weight.new_empty(d_out, rank))
        self.lora_B = nn.Parameter(self.weight.new_zeros(rank, d_in))
        nn.init.normal_(self.lora_A, std=1 / math.sqrt(rank))

    def _row_norm(self, weight):
        # FP32 accumulation avoids half-precision overflow in the reduction.
        if weight.dtype in (torch.float16, torch.bfloat16):
            weight = weight.float()
        return torch.linalg.vector_norm(weight, dim=1, keepdim=True).clamp_min(
            torch.finfo(self.weight.dtype).tiny
        )

    @torch.no_grad()
    def _weight_norm(self):
        # Norm construction needs no autograd graph. Disabling autocast avoids
        # rounding FP32 parameters down while constructing the denominator.
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            weight, a, b = self.weight, self.lora_A, self.lora_B
            if weight.dtype in (torch.float16, torch.bfloat16):
                weight, a, b = weight.float(), a.float(), b.float()
            adapted = a @ b
            adapted.add_(weight)
            return self._row_norm(adapted)

    def forward(self, x):
        # Associativity keeps the dense adapted weight out of the backward
        # graph. Bias is deliberately added after the magnitude rescaling.
        scale = (self.m / self._weight_norm()).flatten()
        output = F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_B), self.lora_A)
        output_dtype = output.dtype
        # Apply the ratio before casting back: the ratio alone may overflow
        # half precision even when the scaled activation is representable.
        output = output * scale
        if self.bias is not None:
            output = output + self.bias
        return output.to(output_dtype)

    @classmethod
    def from_linear(cls, layer, rank=4):
        """Copy a Linear without sharing storage or changing its training mode."""
        if not isinstance(layer, nn.Linear):
            raise TypeError("layer must be an nn.Linear")
        result = cls(layer.in_features, layer.out_features, rank, layer.weight, layer.bias)
        return result.train(layer.training)

    @torch.no_grad()
    def to_linear(self):
        """Export an independent, frozen Linear for inference without adapter overhead."""
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            weight, a, b = self.weight, self.lora_A, self.lora_B
            if weight.dtype in (torch.float16, torch.bfloat16):
                weight, a, b = weight.float(), a.float(), b.float()
            adapted = weight + a @ b
            merged_weight = (adapted * (self.m / self._row_norm(adapted))).to(self.weight.dtype)
        # Meta construction avoids allocating/initializing weights only to replace
        # them, and does not consume the caller's random-number stream.
        result = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device="meta", dtype=self.weight.dtype)
        result.weight = nn.Parameter(merged_weight, requires_grad=False)
        if self.bias is not None:
            result.bias = nn.Parameter(self.bias.detach().clone(), requires_grad=False)
        return result.train(self.training)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, bias={self.bias is not None}")


def replace_linear_with_dora(model, rank=4):
    """Replace Linear modules in place and return the (possibly replaced) root.

    Shared module aliases stay shared. Other kinds of parameters retain their
    existing requires_grad setting; freeze those separately for adapter-only
    fine-tuning of a larger model. Recreate the optimizer after conversion.
    """
    replacements = {}

    def replace(module):
        if id(module) in replacements:
            return replacements[id(module)]
        if isinstance(module, nn.Linear):
            result = DoRALayer.from_linear(module, rank=rank)
            replacements[id(module)] = result
            return result
        replacements[id(module)] = module
        # named_children() deduplicates shared aliases, so visit registrations.
        for name, child in list(module._modules.items()):
            if child is not None:
                setattr(module, name, replace(child))
        return module

    return replace(model)


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer1(x)


def generate_data(num_samples=100, input_dim=10):
    x = torch.randn(num_samples, input_dim)
    y = torch.sum(x, dim=1, keepdim=True)
    return x, y


def train(model, criterion, optimizer, data_loader, epochs=5):
    model.train()
    for _ in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()


def print_model_parameters(model):
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def main():
    # Importing this module must not reset an application's RNG state.
    torch.manual_seed(0)
    input_dim, output_dim = 10, 1
    model = SimpleModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    x, y = generate_data(num_samples=1000, input_dim=input_dim)
    data_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    eval_x, eval_y = generate_data(num_samples=1000, input_dim=input_dim)

    print_model_parameters(model)
    train(model, criterion, optimizer, data_loader, epochs=100)
    model.eval()
    with torch.no_grad():
        before = model(eval_x)
        print(f"Final Evaluation Loss: {criterion(before, eval_y).item()}")

    model = replace_linear_with_dora(model)
    with torch.no_grad():
        torch.testing.assert_close(model(eval_x), before)
    print_model_parameters(model)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    print("Continuing training with DoRA layers...")
    train(model, criterion, optimizer, data_loader, epochs=5)
    model.eval()
    with torch.no_grad():
        print(f"Final (DoRA) Evaluation Loss: {criterion(model(eval_x), eval_y).item()}")


if __name__ == "__main__":
    main()
