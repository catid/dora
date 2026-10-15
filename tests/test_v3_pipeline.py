import itertools
import torch
from torch import nn

from rankstudy.v3.adapters import AdapterLinear,adapted,inject
from rankstudy.v3.compression import actual_factors,thin_svd,balanced_factors,prefix_ranks,compress_modules,checkpoint_modules,write_export
from rankstudy.v3.checkpoint import save_checkpoint,load_adapter
from rankstudy.v3.pipeline import token_boundary,stage_seed,shuffle_compatible,export_policy


def test_actual_masked_updates_svd_all_representations_and_zero_rank():
    torch.manual_seed(3)
    for init in ['L','I','T0','T1','ADALORA']:
        layer=AdapterLinear(nn.Linear(7,6,bias=False),4,init,2.5,5).double()
        with torch.no_grad():
            for p in layer.parameters(recurse=False): p.add_(torch.randn_like(p)*.12)
        layer.set_mask(torch.tensor([True,False,True,False]))
        c,d=actual_factors(layer)
        want=layer.branch(torch.eye(7,dtype=torch.float64)).T
        torch.testing.assert_close(c@d,want,atol=1e-12,rtol=1e-12)
        l,s,r=thin_svd(c,d)
        torch.testing.assert_close((l*s)@r.T,want,atol=1e-12,rtol=1e-12)
        a,b=balanced_factors(l,s,r,len(s),scale=2.5)
        torch.testing.assert_close(2.5*b@a,want,atol=1e-12,rtol=1e-12)
        for rank in range(len(s)+1):
            a,b=balanced_factors(l,s,r,rank,scale=2.5)
            error=(want-2.5*b@a).norm()
            torch.testing.assert_close(error,s[rank:].square().sum().sqrt(),atol=1e-12,rtol=1e-12)
    zero=AdapterLinear(nn.Linear(7,6,bias=False),0,'L',1.).double()
    c,d=actual_factors(zero); l,s,r=thin_svd(c,d)
    assert s.numel()==0 and l.shape==(6,0) and r.shape==(7,0)


def test_prefix_allocation_matches_exhaustive_unequal_costs_and_ties():
    for scores in [{'a':torch.tensor([3.,2.,1.]),'b':torch.tensor([2.5,1.,0.])},
                   {'a':torch.zeros(3),'b':torch.zeros(3)},
                   {'a':torch.ones(3),'b':torch.ones(3)}]:
        for costs in [{'a':2,'b':3},{'a':5,'b':5}]:
            for budget in range(0,19):
                got,info=prefix_ranks(scores,costs,budget)
                feasible=[]
                for a,b in itertools.product(range(4),repeat=2):
                    if a*costs['a']+b*costs['b']<=budget:
                        feasible.append(float(scores['a'][:a].square().sum()+scores['b'][:b].square().sum()))
                value=sum(float(scores[n][:got[n]].square().sum()) for n in scores)
                assert value==max(feasible)
                assert info['spent']<=budget


class Targets(nn.Module):
    def __init__(self):
        super().__init__(); self.q_proj=nn.Linear(7,6,bias=False); self.o_proj=nn.Linear(7,6,bias=False)
    def forward(self,x): return self.q_proj(x)+self.o_proj(x)


def test_standalone_export_loads_on_original_backbone_without_correction_loss(tmp_path):
    torch.manual_seed(82); base=Targets(); original={n:p.clone() for n,p in base.state_dict().items()}
    model=inject(base,3,'I',4.,17)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad: p.add_(torch.randn_like(p)*.2)
    model.q_proj.set_mask(torch.tensor([True,False,True]))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],weight_decay=0.)
    ck=tmp_path/'checkpoint'; save_checkpoint(ck,model,opt,{'probe':True})
    loaded=checkpoint_modules(ck,{'q_proj':(6,7),'o_proj':(6,7)})
    factors,pattern,report=compress_modules(loaded,3,scale=4.,policy='exact')
    export=tmp_path/'export'; write_export(export,factors,4.,report,{'checkpoint':str(ck)})
    fresh=Targets(); fresh.load_state_dict(original)
    fresh=inject(fresh,3,'L',4.,99,pattern=pattern)
    load_adapter(export,fresh,'cpu')
    x=torch.randn(11,7)
    torch.testing.assert_close(fresh(x),model(x),atol=4e-6,rtol=4e-6)
    for name,m in adapted(fresh).items():
        assert torch.equal(m.base.weight,original[name+'.weight'])
        assert not hasattr(m,'A0') and not hasattr(m,'B0')


def test_true_rank_deficiency_and_truncated_budget():
    c=torch.tensor([[1.,1.],[2.,2.],[3.,3.]],dtype=torch.float64)
    d=torch.tensor([[1.,2.,3.,4.],[2.,4.,6.,8.]],dtype=torch.float64)
    l,s,r=thin_svd(c,d)
    torch.testing.assert_close((l*s)@r.T,c@d,atol=1e-12,rtol=1e-12)
    assert s[1]<1e-12
    model=inject(Targets(),4,'I',1.,2)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad: p.add_(torch.randn_like(p)*.1)
    _,ranks,report=compress_modules(adapted(model),2,policy='global')
    assert sum(ranks.values())==4 and report['allocation']['spent']==4*13


def test_token_pipeline_boundaries_and_seed_stream_independence():
    updates=[{'input_tokens':x} for x in [9,12,10,11,8,12,10,9,12,8,7,8]]
    pilot,pr=token_boundary(updates,.20); source,cr=token_boundary(updates,.75)
    assert 0<pilot<source<len(updates)
    assert pr['overshoot_tokens']<max(x['input_tokens'] for x in updates)
    assert cr['overshoot_tokens']<max(x['input_tokens'] for x in updates)
    # No training example IDs can be duplicated by disjoint stage slices.
    assert list(range(pilot))+list(range(pilot,source))+list(range(source,len(updates)))==list(range(len(updates)))
    assert stage_seed(1001,'pilot')!=stage_seed(1001,'student')
    assert stage_seed(1001,'student')==stage_seed(1001,'student')


def test_fresh_student_only_uses_pattern_and_compatible_shuffle():
    pilot=inject(Targets(),4,'I',1.,stage_seed(1001,'pilot'))
    pilot.q_proj.set_mask(torch.tensor([True,False,False,True]))
    pilot.o_proj.set_mask(torch.tensor([True,False,True,True]))
    pattern={n:int(m.mask.sum()) for n,m in adapted(pilot).items()}
    student=inject(Targets(),3,'I',1.,stage_seed(1001,'student'),pattern=pattern)
    for n,m in adapted(student).items():
        assert m.U.count_nonzero()==0 and m.V.count_nonzero()==0 and m.mask.all()
        assert not torch.equal(m.A0,pilot.get_submodule(n).A0[:m.rank])
    pattern={'a':1,'b':3,'c':2,'d':4}
    shapes={'a':(2,3),'b':(2,3),'c':(3,2),'d':(3,2)}
    shuffled=shuffle_compatible(pattern,shapes,4)
    assert sorted([shuffled['a'],shuffled['b']])==[1,3]
    assert sorted([shuffled['c'],shuffled['d']])==[2,4]


def test_common_export_global_energy_fills_same_budget_for_nonuniform_sources(tmp_path):
    """A narrow module must not silently shrink the whole deployment budget."""
    for init in ['L','I','T0','T1','ADALORA']:
        model=inject(Targets(),4,init,1.,42,pattern={'q_proj':1,'o_proj':4})
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad: p.add_(torch.randn_like(p)*.1)
        factors,ranks,report=compress_modules(adapted(model),2,policy='global')
        assert sum(ranks.values())==4
        assert report['allocation']['spent']==52
        assert report['allocation']['underspend']==0
        assert report['allocation']['algorithm']=='equal_cost_exact_global_energy'
        path=tmp_path/init
        write_export(path,factors,1.,report,{'test_only':True})
        import json
        desc=json.loads((path/'manifest.json').read_text())
        assert desc['ordinary_factor_scalars']==52
        assert desc['ordinary_factor_bytes']==208
    for method in ['L_FIXED','I_GATE','ADALORA_REF','T0_GATE','T1_GATE',
                   'L_PILOT_I_STUDENT','L_UNIFORM_SVD','L_GLOBAL_SVD']:
        assert export_policy(method,True)=='global'
    assert export_policy('L_UNIFORM_SVD')=='uniform'
    assert export_policy('I_GLOBAL_SVD')=='global'
