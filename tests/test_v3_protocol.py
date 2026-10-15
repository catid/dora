"""Protocol tests exercise executed code, including the previous v2 defects."""
import itertools
import copy

import pytest
import torch
from torch import nn

from rankstudy.v3.adapters import AdapterLinear, compact
from rankstudy.v3.allocation import Candidate, TokenSchedule, allocate
from rankstudy.v3.scoring import SlotEMA, token_subsample
from rankstudy.v3.reference import reference_allocator, reference_after_step, reference_penalty
from rankstudy.v3.vendor.adalora.adalora import SVDLinear, RankAllocator, compute_orth_regu
from rankstudy.v3.checkpoint import save_checkpoint, load_checkpoint
from scripts.run_one_v3 import numerator


def layer(init, score="none", rank=3, dtype=torch.float64):
    return AdapterLinear(nn.Linear(5, 4, bias=False), rank, init, 1.7, 42, score).to(dtype)


def direct(m, x, gate):
    mask = m.mask*gate
    if m.init == "L":
        return ((x @ m.A.T)*mask) @ m.B.T * m.scale
    if m.init == "I":
        return (((x @ (m.A0+m.U).T)*mask) @ (m.B0+m.V).T
                - ((x @ m.A0.T)*mask) @ m.B0.T)*m.scale
    q, p = m.factors()
    lam = m.lambda0 + m.dlambda
    return (((x @ q.T)*lam*mask) @ p.T
            - ((x @ m.Q0.T)*m.lambda0*mask) @ m.P0.T)*m.scale


@pytest.mark.parametrize("init", ["L", "I", "T0", "T1"])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-12), (torch.float32, 3e-5)])
def test_expanded_outputs_and_task_gradients_all_masks(init, dtype, tol):
    torch.manual_seed(81)
    m = layer(init, dtype=dtype)
    x = torch.randn(6, 5, dtype=dtype, requires_grad=True)
    p = tuple(m.parameters(recurse=False))
    for bits in itertools.product([False, True], repeat=3):
        m.mask.copy_(torch.tensor(bits))
        assert torch.count_nonzero(m.branch(x)) == 0
    with torch.no_grad():
        for v in p:
            v.add_(torch.randn_like(v)*.1)
    for bits in itertools.product([False, True], repeat=3):
        m.mask.copy_(torch.tensor(bits))
        gate = torch.ones(3, dtype=dtype, requires_grad=True)
        a, b = m.branch(x, gate), direct(m, x, gate)
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
        ga = torch.autograd.grad(a.square().sum(), (*p, x, gate), retain_graph=True)
        gb = torch.autograd.grad(b.square().sum(), (*p, x, gate))
        for va, vb in zip(ga, gb):
            torch.testing.assert_close(va, vb, atol=tol, rtol=tol)


@pytest.mark.parametrize("init", ["L", "I", "T0", "T1"])
def test_expected_initial_gradients(init):
    m = layer(init)
    m.branch(torch.randn(8, 5, dtype=torch.float64)).square().sum().backward()
    # A nonzero target is necessary; the centered zero function's squared norm
    # has zero task gradients for every method, so use a linear loss instead.
    m.zero_grad()
    m.branch(torch.ones(8, 5, dtype=torch.float64)).sum().backward()
    nz = {n: bool(p.grad.count_nonzero()) for n, p in m.named_parameters(recurse=False)}
    expected = {"L": {"A": False, "B": True}, "I": {"U": True, "V": True},
                "T0": {"dP": False, "dQ": False, "dlambda": True},
                "T1": {"dP": True, "dQ": True, "dlambda": True}}
    assert nz == expected[init]


def test_displacement_retains_sub_ulp_steps():
    m = layer("I", dtype=torch.float32)
    with torch.no_grad():
        m.A0.fill_(1.)
        m.U.fill_(1e-9)
    assert torch.equal(m.A0+m.U, m.A0)  # Direct factors would lose this step.
    assert m.branch(torch.ones(2, 5)).abs().sum() > 0


@pytest.mark.parametrize("init", ["L", "I", "T0", "T1"])
def test_gate_grad_accumulation_matches_dense_deletion_and_microbatches(init):
    torch.manual_seed(28)
    m = layer(init, "GATE")
    with torch.no_grad():
        for p in m.parameters(recurse=False):
            p.add_(torch.randn_like(p)*.15)
    x, target = torch.randn(7, 5, dtype=torch.float64), torch.randn(7, 4, dtype=torch.float64)
    scores = []
    for chunks in [[slice(None)], [slice(0, 2), slice(2, 6), slice(6, 7)]]:
        m.zero_grad(); m.begin_update()
        for sl in chunks:
            (m(x[sl])-target[sl]).square().sum().backward()
        scores.append(m.task_score(28))
    torch.testing.assert_close(scores[0], scores[1], atol=1e-12, rtol=1e-12)
    eps = 1e-6
    for i in range(3):
        g = torch.ones(3, dtype=torch.float64)
        g[i] += eps
        hi = (m.base(x)+m.branch(x, g)-target).square().mean()
        g[i] -= 2*eps
        lo = (m.base(x)+m.branch(x, g)-target).square().mean()
        torch.testing.assert_close(scores[0][i], ((hi-lo)/(2*eps)).abs(), atol=1e-8, rtol=1e-8)
    assert all(m._gate is not p for p in m.parameters())


@pytest.mark.parametrize("init", ["L", "I"])
def test_act_gram_identity_and_padding_independence(init):
    torch.manual_seed(2)
    m = layer(init, "ACT")
    with torch.no_grad():
        for p in m.parameters(recurse=False):
            p.add_(torch.randn_like(p)*.1)
    x = torch.randn(15, 5, dtype=torch.float64)
    got = m.activation_energy(x)
    expected = []
    for i in range(3):
        gate = torch.zeros(3, dtype=torch.float64); gate[i] = 1
        expected.append(m.branch(x, gate).square().sum())
    torch.testing.assert_close(got, torch.stack(expected), atol=1e-11, rtol=1e-11)
    masks = [torch.tensor([[1, 1, 0], [1, 1, 1]]), torch.tensor([[1, 0]])]
    idx = token_subsample(masks, 33, 2, 4)
    assert sum(len(i) for i in idx) == 4
    assert all(bool(m.reshape(-1)[i].all()) for m, i in zip(masks, idx))
    # Same valid-token universe, different partitioning and padding.
    other = token_subsample([torch.ones(6, dtype=torch.long)], 33, 2, 4)[0]
    flat = []; offset = 0
    for mask, ind in zip(masks, idx):
        valid = torch.nonzero(mask.flatten()).flatten().tolist()
        flat += [offset + valid.index(int(i)) for i in ind]
        offset += len(valid)
    assert flat == other.tolist()


def test_raw_uses_full_factors_and_disp_uses_displacements():
    m = layer("I", "RAW")
    m.U.grad = torch.ones_like(m.U); m.V.grad = torch.ones_like(m.V)
    assert m.task_score().sum() > 0
    m.score = "DISP"
    assert m.task_score().sum() == 0


def test_every_update_ema_and_uncertainty_not_only_allocation_steps():
    ema = SlotEMA(uncertainty=True)
    ref, unc = torch.zeros(2), torch.zeros(2)
    for step in range(20):
        q = torch.tensor([float(step), float(20-step)])
        ema.update({"x": q})
        ref = .85*ref + .15*q
        unc = .85*unc + .15*(q-ref).abs()
        torch.testing.assert_close(ema.value("x"), ref*unc)
    assert ema.updates == 20


def test_token_schedule_force_boundary_and_recovery():
    counts = [23, 1, 4, 2, 3, 7, 4, 2, 3, 4, 1, 1, 8, 7, 4, 5, 9, 1, 2, 2]
    schedule = TokenSchedule(counts, 1000, 400)
    tokens, events = 0, []
    for step, count in enumerate(counts, 1):
        tokens += count
        budget, event, freeze = schedule.at(step, tokens)
        if event:
            events.append((step, budget))
        if freeze:
            assert event and budget == 400
    assert events[-1][0] == schedule.freeze_update < len(counts)


def test_allocator_against_exhaustive_heterogeneous_floors_ties():
    for values in [[0]*7, [1]*7, [4, 0, 1, 7, 2, 1, 8]]:
        items = [Candidate("a" if i<3 else "b", i, 2 if i<3 else 3, values[i]) for i in range(7)]
        for budget in range(5, 19):
            picked, receipt = allocate(items, budget, floor=1)
            feasible = []
            for bits in itertools.product([False, True], repeat=7):
                use = [c for c, b in zip(items, bits) if b]
                cost = sum(c.cost for c in use)
                if cost <= budget and {c.module for c in use} == {"a", "b"}:
                    feasible.append(sum(c.value for c in use))
            value = sum(c.value for c in items if (c.module, c.slot) in picked)
            assert value == max(feasible)
            assert receipt['spent'] <= budget
            assert allocate(list(reversed(items)), budget, 1)[0] == picked


@pytest.mark.parametrize("init", ["L", "I", "T0", "T1"])
def test_no_momentum_drift_compaction_and_checkpoint_parity(init):
    torch.manual_seed(6)
    m = nn.Sequential(layer(init))
    a = m[0]
    original_base = a.base.weight.clone()
    origins = {n: p.clone() for n,p in a.named_buffers() if n.endswith('0')}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.02, weight_decay=0.)
    x = torch.randn(6, 5, dtype=torch.float64)
    for i in range(3):
        opt.zero_grad(); m(x).square().sum().backward(); opt.step()
    before = {n: p.detach().clone() for n,p in a.named_parameters(recurse=False)}
    a.set_mask(torch.tensor([True, False, True]), opt)
    opt.zero_grad(); m(x).square().sum().backward(); a.freeze_pruned(opt); opt.step()
    for name,p in a.named_parameters(recurse=False):
        axis=a.axes()[name]
        torch.testing.assert_close(p.select(axis,1), before[name].select(axis,1), rtol=0, atol=0)
    want = m(x).detach()
    state_before = {n: opt.state[p]['exp_avg'].clone() for n,p in a.named_parameters(recurse=False)}
    # Compaction must preserve dtype as well as the survivor values.
    newopt = compact(m,opt)
    torch.testing.assert_close(m(x), want, rtol=1e-12, atol=1e-12)
    assert m[0].slot_ids.tolist() == [0,2]
    for name,p in m[0].named_parameters(recurse=False):
        expected=state_before[name].index_select(m[0].axes()[name], torch.tensor([0,2]))
        torch.testing.assert_close(newopt.state[p]['exp_avg'], expected, rtol=0, atol=0)
    assert torch.equal(m[0].base.weight, original_base)
    for name,v in origins.items():
        torch.testing.assert_close(getattr(m[0],name),v.index_select(a.axes()[name],torch.tensor([0,2])),rtol=0,atol=0)
    assert float(m[0].scale) == 1.7


def test_reference_matches_upstream_full_optimizer_trajectory_and_final_freeze():
    torch.manual_seed(9)
    current = nn.Sequential(layer('ADALORA', rank=4))
    upstream = nn.Sequential(SVDLinear(5,4,r=4,lora_alpha=1.7,bias=False,merge_weights=False).double())
    with torch.no_grad():
        upstream[0].weight.copy_(current[0].base.weight)
        for name in ['lora_A','lora_B','lora_E']:
            getattr(upstream[0],name).copy_(getattr(current[0],name))
    ours = reference_allocator(current,4,2,40)
    ref = RankAllocator(upstream,4,2,4,8,16,.85,.85,total_step=40)
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.02,weight_decay=0.) for m in (current,upstream)]
    for step in range(1,41):
        x,y = torch.randn(8,5,dtype=torch.float64),torch.randn(8,4,dtype=torch.float64)
        for m,opt in zip((current,upstream),optimizers):
            opt.zero_grad()
            penalty = reference_penalty(m,.1) if m is current else compute_orth_regu(m,.1)
            ((m(x)-y).square().mean()+penalty).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
        reference_after_step(current,ours,step)
        ref.update_and_mask(upstream,step)
        for name in ['lora_A','lora_B','lora_E']:
            torch.testing.assert_close(getattr(current[0],name),getattr(upstream[0],name),atol=2e-12,rtol=2e-12)
        for name,state in ours.exp_avg_ipt.items():
            torch.testing.assert_close(state,ref.exp_avg_ipt[name],atol=2e-12,rtol=2e-12)
        if step > 32:
            assert current[0].lora_E.count_nonzero() <= 2


def test_safe_checkpoint_resume_matches_optimizer_rng_masks_and_scores(tmp_path):
    torch.manual_seed(89)
    model=nn.Sequential(layer('I','GATE'))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0.)
    ema=SlotEMA()
    def update(m,o,e):
        o.zero_grad(); m[0].begin_update()
        m(torch.randn(3,5,dtype=torch.float64)).sum().backward()
        e.update({'x':m[0].task_score()})
        m[0].freeze_pruned(o); o.step()
    update(model,opt,ema)
    model[0].set_mask(torch.tensor([True,False,True]),opt)
    path=tmp_path/'checkpoint'
    save_checkpoint(path,model,opt,{'ema':ema.state_dict(),'step':1})
    update(model,opt,ema)
    expected={n:v.clone() for n,v in model.state_dict().items()}
    restored=nn.Sequential(layer('I','GATE'))
    restored[0].base.load_state_dict(model[0].base.state_dict())
    newopt=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.01,weight_decay=0.)
    extra=load_checkpoint(path,restored,newopt,'cpu')
    e=SlotEMA(); e.load_state_dict(extra['ema'])
    update(restored,newopt,e)
    for name,v in restored.state_dict().items():
        torch.testing.assert_close(v,expected[name],rtol=0,atol=0)
    torch.testing.assert_close(e.value('x'),ema.value('x'),rtol=0,atol=0)


def test_loss_flattening_preserves_masking_shift_and_batch_mean():
    torch.manual_seed(33)
    logits=torch.randn(3,7,11,dtype=torch.float64,requires_grad=True)
    labels=torch.randint(11,(3,7)); labels[:,0:3]=-100; labels[1,5:]=-100
    total,count=numerator(logits,labels)
    manual=sum(-logits[i,j-1].float().log_softmax(-1)[labels[i,j]] for i in range(3) for j in range(1,7) if labels[i,j]!=-100)
    assert count==sum(int(labels[i,j]!=-100) for i in range(3) for j in range(1,7))
    torch.testing.assert_close(total,manual)
    ga=torch.autograd.grad(total/count,logits,retain_graph=True)[0]
    gb=torch.autograd.grad(manual/count,logits)[0]
    torch.testing.assert_close(ga,gb)
