import torch
from torch import nn

from rankstudy.compress import thin_core_svd
from scripts.run_pipeline_v2 import export_factors, inject_pattern


class TinyTargets(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(6, 5, bias=False)
        self.o_proj = nn.Linear(6, 5, bias=False)


def test_pattern_injection_uses_fresh_per_module_ranks():
    torch.manual_seed(4)
    model = inject_pattern(TinyTargets(), {"q_proj": 2, "o_proj": 3}, "L", 1.0,
                           "TEST", 9, target_modules=("q_proj", "o_proj"))
    assert model.q_proj.rank == 2
    assert model.o_proj.rank == 3
    assert model.q_proj.active_mask.all()


def test_pipeline_export_factors_reconstructs_thin_core_update():
    torch.manual_seed(5)
    d, n, r = 5, 6, 3
    a0 = torch.randn(r, n)
    b0 = torch.randn(d, r)
    a = a0 + .1 * torch.randn_like(a0)
    b = b0 + .1 * torch.randn_like(b0)
    left, s, right, meta = thin_core_svd(a0, b0, a, b, scale=1.7)
    factors = export_factors([("x", None, left, s, right, meta)], {"x": len(s)})
    reconstructed = factors["x"]["B"] @ factors["x"]["A"]
    direct = 1.7 * (b @ a - b0 @ a0)
    assert torch.allclose(reconstructed, direct, atol=2e-5, rtol=2e-5)
