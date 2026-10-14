import torch
from rankstudy.adapters import StudyLinear
from rankstudy.algebra import FaithfulAdaLoRA
from scripts.run_one import loss_fn

def test_standard_zero_start_and_learning_gradient():
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.bfloat16)
    m=StudyLinear(base,rank=3,init='L',scale=1.)
    x=torch.randn(2,5); y=m(x); assert torch.allclose(y,base(x.to(torch.bfloat16)),atol=1e-2,rtol=1e-2)
    y.float().sum().backward(); assert m.B.grad is not None and torch.count_nonzero(m.B.grad)>0

def test_inilora_centered_identity_and_nonzero_gradients():
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.bfloat16)
    m=StudyLinear(base,rank=3,init='I',scale=1.)
    x=torch.randn(2,5); assert torch.allclose(m(x),base(x.to(torch.bfloat16)),atol=1e-2,rtol=1e-2)
    m(x).float().sum().backward(); assert torch.count_nonzero(m.A.grad)>0 and torch.count_nonzero(m.B.grad)>0

def test_mask_removes_centered_component():
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.bfloat16); m=StudyLinear(base,3,init='I')
    m.A.data.add_(.1); m.B.data.add_(.2); m.set_mask(torch.zeros(3,dtype=torch.bool)); assert torch.count_nonzero(m.adapter.delta_matrix())==0

def test_triplet_start_is_zero_and_masked():
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.bfloat16); m=StudyLinear(base,3,init='T1',triplet=True)
    x=torch.randn(2,5); assert torch.allclose(m(x),base(x.to(torch.bfloat16)),atol=1e-2,rtol=1e-2)
    m.P.data.add_(.1); m.set_mask(torch.zeros(3,dtype=torch.bool)); assert torch.count_nonzero(m.adapter.delta_expanded())==0

def test_adalora_reference_initialization_and_scale():
    torch.manual_seed(4)
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.bfloat16)
    m=StudyLinear(base,rank=3,init='ADALORA',scale=2.0,adalora=True)
    assert m.adapter.E.shape == (3,1)
    assert torch.count_nonzero(m.adapter.E) == 0
    assert torch.allclose(m.adapter.ranknum, torch.tensor(3., dtype=torch.float32))
    x=torch.randn(2,5)
    assert torch.allclose(m(x),base(x.to(torch.bfloat16)),atol=1e-2,rtol=1e-2)
    with torch.no_grad(): m.E[:,0].fill_(1.)
    expected=(m.B @ m.A) * (2.0/(3.0+1e-5))
    assert torch.allclose(m.adapter.delta_direct(), expected, atol=1e-6, rtol=1e-6)

def test_adalora_mask_removes_whole_slot_and_e_gradients():
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.bfloat16)
    m=StudyLinear(base,3,init='ADALORA',adalora=True)
    with torch.no_grad():
        m.E[:,0].fill_(1.)
    m.set_mask(torch.tensor([True,False,True]))
    assert torch.count_nonzero(m.adapter.slot_deltas()[1]) == 0
    m(torch.randn(2,5)).float().sum().backward()
    assert m.E.grad is not None and torch.count_nonzero(m.E.grad[[0,2]]) > 0

def test_adalora_masked_slot_can_reactivate_between_reference_allocations():
    torch.manual_seed(41)
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.float32)
    m=StudyLinear(base,rank=3,init='ADALORA',adalora=True)
    m.set_mask(torch.tensor([False, True, True]))
    opt=torch.optim.AdamW([m.E, m.A, m.B],lr=.05,weight_decay=0.)
    m(torch.randn(8,5)).sum().backward(); opt.step()
    # Upstream masks E at an allocation event, then permits the optimizer to
    # move it until the next event.  A persistent pre-forward zero would make
    # the faithful reference arm permanently monotonic instead.
    assert m.E[0,0].abs() > 0
    assert not bool(m.active_mask[0])

def test_loss_fn_is_supervised_token_mean():
    torch.manual_seed(12)
    logits=torch.randn(2,4,7,requires_grad=True)
    labels=torch.tensor([[1,2,-100,-100],[3,4,5,-100]])
    got=loss_fn(logits,labels)
    manual=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,7),labels[:,1:].reshape(-1),ignore_index=-100)
    assert torch.allclose(got,manual)

def test_frozen_base_and_initial_factors_survive_optimizer_step():
    torch.manual_seed(13)
    base=torch.nn.Linear(5,4,bias=False,dtype=torch.float32)
    m=StudyLinear(base,3,init='I'); base_w=base.weight.detach().clone(); a0=m.A0.detach().clone(); b0=m.B0.detach().clone()
    opt=torch.optim.AdamW([m.A,m.B],lr=.01,weight_decay=0.)
    m(torch.randn(4,5)).sum().backward(); opt.step(); m.set_mask(torch.tensor([True,False,True]))
    assert torch.equal(base.weight,base_w)
    assert torch.equal(m.A0,a0) and torch.equal(m.B0,b0)
