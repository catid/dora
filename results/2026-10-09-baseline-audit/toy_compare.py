"""Original-axis/full-gradient versus row-axis ablations on matched held-out data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/var/tmp/dora-bench/baseline_audit')
REPO = Path('/home/catid/dora')
sys.path.insert(0, str(REPO))
from dora import DoRALayer

spec = importlib.util.spec_from_file_location('historical', ROOT / 'original_dora.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
torch.set_num_threads(1)


class Dense(nn.Module):
    def __init__(self, base, a, b, axis=1, detach=False, magnitude_only=False, lora=False):
        super().__init__()
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(a.clone(), requires_grad=not magnitude_only)
        self.lora_B = nn.Parameter(b.clone(), requires_grad=not magnitude_only)
        self.axis, self.detach, self.lora = axis, detach, lora
        if not lora:
            self.m = nn.Parameter(self.weight.norm(dim=axis, keepdim=True))

    def forward(self, x):
        v = self.weight + self.lora_A @ self.lora_B
        if not self.lora:
            norm = v.norm(dim=self.axis, keepdim=True)
            if self.detach:
                norm = norm.detach()
            v = self.m * (v / norm)
        return F.linear(x, v, self.bias)


def fit(model, x, y, epochs, shuffle_seed):
    generator = torch.Generator().manual_seed(shuffle_seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    start = time.perf_counter()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(model(bx), by).backward()
            optimizer.step()
    return time.perf_counter() - start


rows = []
names = ('original_dim0_fullgrad', 'original_magnitude_only', 'row_dim1_fullgrad',
         'row_dim1_detached_dense', 'current_factorized', 'lora', 'continue_full_training')
for seed in range(5):
    torch.manual_seed(seed)
    base = nn.Linear(10, 1)
    x = torch.randn(1000, 10)
    y = x.sum(1, keepdim=True)
    generator = torch.Generator().manual_seed(10000 + seed)
    tx = torch.randn(10000, 10, generator=generator)
    ty = tx.sum(1, keepdim=True)
    fit(base, x, y, 100, 30000 + seed)
    before = F.mse_loss(base(tx), ty).item()
    torch.manual_seed(1000 + seed)
    historical = original.DoRALayer(10, 1, rank=4, weight=base.weight.detach().clone(), bias=base.bias.detach().clone())
    a, b = historical.lora_A.detach().clone(), historical.lora_B.detach().clone()
    for name in names:
        if name == 'original_dim0_fullgrad':
            model = copy.deepcopy(historical)
        elif name == 'current_factorized':
            model = DoRALayer.from_linear(base, rank=4)
            with torch.no_grad():
                model.lora_A.copy_(a)
                model.lora_B.copy_(b)
        elif name == 'continue_full_training':
            model = copy.deepcopy(base)
        else:
            model = Dense(base, a, b, axis=0 if name == 'original_magnitude_only' else 1,
                          detach=name == 'row_dim1_detached_dense',
                          magnitude_only=name == 'original_magnitude_only', lora=name == 'lora')
        torch.testing.assert_close(model(tx), base(tx), atol=1e-6, rtol=1e-5)
        model.zero_grad(set_to_none=True)
        F.mse_loss(model(x[:64]), y[:64]).backward()
        initial_grad = {n: p.grad.norm().item() if p.grad is not None else None for n, p in model.named_parameters()}
        start_state = {n: p.detach().clone() for n, p in model.named_parameters()}
        seconds = fit(model, x, y, 5, 20000 + seed)
        with torch.no_grad():
            loss = F.mse_loss(model(tx), ty).item()
            assert torch.isfinite(torch.tensor(loss))
        row = {'seed': seed, 'method': name, 'heldout_mse_before': before, 'heldout_mse_after': loss,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'initial_gradient_norms': initial_grad,
               'parameter_change_norms': {n: (p.detach() - start_state[n]).norm().item() for n, p in model.named_parameters()},
               'training_seconds': seconds}
        rows.append(row)
    print(json.dumps({'seed': seed, 'before': before, 'after': {r['method']: r['heldout_mse_after'] for r in rows if r['seed'] == seed}}), flush=True)

summary = {}
for name in names:
    values = [r['heldout_mse_after'] for r in rows if r['method'] == name]
    summary[name] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'values': values,
                     'trainable_parameters': next(r['trainable_parameters'] for r in rows if r['method'] == name)}
data = {'torch': torch.__version__, 'device': 'cpu', 'threads': 1, 'seeds': list(range(5)),
        'original_commit': 'bb97617a0d5e1ad4c6856cb7278f5a7386820d18',
        'protocol': '1000 train, independent10000 heldout examples, y=sum(x). Shared100-epoch pretrained nn.Linear per seed; then5epochs,80AdamW updates, lr.001,defaultdecay.01. Same stored base,A,B and shuffled minibatches for all adapter variants. No LR tuning. Original forward untouched; column-magnitude-only freezes factors. Factorization compares same detached row formula; full gradient row ablation changes only axis vs original. This toy is not downstream task evidence.',
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT/'original_dora.py', REPO/'dora.py')},
        'baseline_heldout_mse': {'mean': statistics.mean(r['heldout_mse_before'] for r in rows[::len(names)]), 'values': [r['heldout_mse_before'] for r in rows[::len(names)]]},
        'summary': summary, 'runs': rows}
(ROOT / 'toy_comparison.json').write_text(json.dumps(data, indent=2)+'\n')
print(json.dumps(summary, indent=2))
