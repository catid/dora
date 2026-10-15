import torch
import yaml
from pathlib import Path
from scripts.synthetic_v3 import problem,oracle,experiment,TEACHER_RANKS
from scripts.synthetic_v3 import EightMatrices
from rankstudy.v3.checkpoint import load_adapter


def test_teacher_rank_spectrum_and_covariance_oracle_are_known():
    torch.set_num_threads(1)
    for covariance in ['isotropic','anisotropic']:
        data=problem(901,covariance,False)
        for delta,rank in zip(data['teacher'],TEACHER_RANKS):
            s=torch.linalg.svdvals(delta)
            assert int((s>1e-5).sum())==rank
            torch.testing.assert_close(s[:rank],torch.linspace(1.,.1,rank),atol=5e-7,rtol=5e-7)
        small=oracle(data,16*32);exact=oracle(data,32*32)
        assert small['dev_function_mse']>1e-5
        assert exact['dev_function_mse']<1e-10
        noisy=problem(901,covariance,True)
        assert torch.equal(noisy['train'],data['train']) and torch.equal(noisy['dev'],data['dev'])
        assert not torch.equal(data['train'],data['dev'][:,:128])
        noise_error=oracle(noisy,32*32)['dev_observed_mse']
        assert abs(noise_error-.03**2)<3e-5


def test_synthetic_restart_and_compression_use_same_exposure_and_exact_budget():
    specs={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    hashes=[]
    for method in ['L_FIXED','I_GATE','I_PILOT_L_STUDENT','I_GLOBAL_SVD']:
        r=experiment(method,901,'isotropic',False,2,32,specs)
        assert sum(s['updates'] for s in r['stages'])==32
        assert r['active_scalars']<=r['P_target']
        assert r['backbone_unchanged']
        hashes.append(r['data_sha256'])
    assert len(set(hashes))==1


def test_retained_synthetic_pipeline_reloads_on_original_matrices(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    specs={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    r=experiment('I_GLOBAL_SVD',901,'isotropic',False,2,32,specs,tmp_path/'checkpoints')
    path=Path(r['stages'][-1]['checkpoint']);desc=json.loads((path/'manifest.json').read_text())
    pattern={n:m['rank'] for n,m in desc['modules'].items()}
    data=problem(901,'isotropic',False)
    fresh=EightMatrices(data['base'],'L',pattern,999)
    load_adapter(path,fresh,'cpu')
    with torch.no_grad():loss=float((fresh(data['dev'])-data['dev_clean']).square().mean())
    assert loss==r['dev_function_mse']
    state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
    assert state['optimizer']['state'] and state['extra']['global_update_offset']==24
    assert state['cuda_rng']==[]
