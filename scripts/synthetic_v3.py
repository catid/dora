#!/usr/bin/env python3
"""All-family synthetic control using the corrected adapter/allocator code.

Fixed teacher construction, independent train/dev draws, three feasible budgets,
isotropic/anisotropic covariance, and noisy/clean targets. This is a common-config
mechanistic control, not a tuned algorithm leaderboard or an LLM HPO input.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
import yaml
from rankstudy.v3.adapters import AdapterLinear,adapted,compact
from rankstudy.v3.allocation import Candidate,allocate,uniform_allocate,TokenSchedule
from rankstudy.v3.scoring import SlotEMA
from rankstudy.v3.reference import reference_allocator,reference_after_step,reference_penalty
from rankstudy.v3.compression import compress_modules,actual_factors,prefix_ranks
from rankstudy.v3.pipeline import stage_seed,shuffle_compatible
from rankstudy.v3.checkpoint import json_write,save_checkpoint


TEACHER_RANKS=[0,0,1,1,2,4,8,16]


def problem(seed,covariance,noise):
    g=torch.Generator().manual_seed(20261014)
    base=[];teacher=[];transforms=[]
    for r in TEACHER_RANKS:
        base.append(torch.randn(16,16,generator=g)*.1)
        u,_=torch.linalg.qr(torch.randn(16,16,generator=g))
        v,_=torch.linalg.qr(torch.randn(16,16,generator=g))
        teacher.append((u[:,:r]*torch.linspace(1.,.1,r))@v[:,:r].T if r else torch.zeros(16,16))
        rot,_=torch.linalg.qr(torch.randn(16,16,generator=g))
        transforms.append(rot@torch.diag(torch.linspace(3.,.3,16))@rot.T if covariance=='anisotropic' else torch.eye(16))
    base,teacher,chol=map(torch.stack,[base,teacher,transforms])
    draw=torch.Generator().manual_seed(seed)
    train=torch.randn(8,128,16,generator=draw)@chol.transpose(1,2)
    dev=torch.randn(8,512,16,generator=draw)@chol.transpose(1,2)
    targets=train@(base+teacher).transpose(1,2)
    if noise:targets+=.03*torch.randn(targets.shape,generator=draw)
    dev_clean=dev@(base+teacher).transpose(1,2)
    dev_noisy=dev_clean+(.03*torch.randn(dev_clean.shape,generator=draw) if noise else 0.)
    return dict(base=base,teacher=teacher,chol=chol,train=train,targets=targets,dev=dev,
                dev_clean=dev_clean,dev_noisy=dev_noisy,teacher_seed=20261014,data_seed=seed)


class EightMatrices(nn.Module):
    def __init__(self,base,init,pattern,seed,score='none',scale=1.):
        super().__init__();self.layers=nn.ModuleList()
        for i,w in enumerate(base):
            layer=nn.Linear(16,16,bias=False)
            with torch.no_grad():layer.weight.copy_(w)
            self.layers.append(AdapterLinear(layer,pattern[f'layers.{i}'],init,scale,seed+i*1009,score))
    def forward(self,x):return torch.stack([m(x[i]) for i,m in enumerate(self.layers)])


def lr_at(step,total,peak=.01):
    warm=max(1,round(.03*total))
    if step<warm:return peak*(step+1)/warm
    return peak*(.1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1))))


def fit(data,spec,k,seed,offset,length,stop=None,model=None,pattern=None,checkpoint_path=None):
    init=spec['init'];online=spec['allocator']!='none' and not spec['adalora']
    rank=2*k if online or spec['adalora'] or spec['group']=='wide' else k
    budget=8*k*32
    if model is None:
        if pattern is None:pattern={f'layers.{i}':rank for i in range(8)}
        if spec['triplet'] and not online:
            keep,_=uniform_allocate([Candidate(n,j,33,0.) for n in pattern for j in range(rank)],budget,1)
            pattern={n:sum(x==n for x,_ in keep) for n in pattern}
        model=EightMatrices(data['base'],init,pattern,seed,spec['score'],float(2*k) if spec['adalora'] else 1.)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0.)
    ema=SlotEMA(uncertainty='UNCERTAINTY' in spec['id']);compacted=False
    schedule=TokenSchedule([128]*length,sum(m.rank*m.slot_cost for m in adapted(model).values()),budget)
    ref=reference_allocator(model,2*k,k,length) if spec['adalora'] else None
    history=[];events=[];begin=time.perf_counter();initial=float((model(data['dev'])-data['dev_clean']).square().mean().detach())
    for step in range(stop or length):
        mods=adapted(model);tracking=online and not compacted
        for m in mods.values():
            m.begin_update(tracking and spec['allocator']!='global_random')
            m._sample_indices=torch.arange(128) if tracking and spec['score']=='ACT' else None
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=lr_at(step,length)
        # Identical fixed training draw at every update; no changing exposure.
        task=(model(data['train'])-data['targets']).square().mean();task.backward()
        cost,event,freeze=schedule.at(step+1,(step+1)*128)
        if tracking:
            scores={n:(torch.tensor([random.Random(f'{seed}/{n}/{int(i)}').random() for i in m.slot_ids])
                       if spec['allocator']=='global_random' else m.task_score(1)) for n,m in mods.items()}
            ema.update(scores)
            if event:
                cells=[Candidate(n,int(i),m.slot_cost,float(q)) for n,m in mods.items()
                       for i,q,active in zip(m.slot_ids,ema.value(n),m.mask) if active]
                keep,info=(uniform_allocate if spec['allocator']=='uniform_gate' else allocate)(cells,cost,1)
                for n,m in mods.items():m.set_mask(torch.tensor([(n,int(i)) in keep for i in m.slot_ids]),opt)
                events.append({'update':offset+step+1,**info})
        reg=(reference_penalty(model,.1) if ref is not None else
             .001*torch.stack([m.penalty() for m in mods.values()]).mean() if spec['penalty'] else torch.tensor(0.))
        if reg.requires_grad:reg.backward()
        for m in mods.values():m.freeze_pruned(opt)
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.))
        opt.step()
        if ref is not None:reference_after_step(model,ref,step+1)
        if online and freeze:opt=compact(model,opt);compacted=True
        history.append({'update':offset+step+1,'task_mse':float(task.detach()),'regularizer':float(reg.detach()),
                        'gradient_norm':norm,'active_scalars':sum(int(m.mask.sum())*m.slot_cost for m in adapted(model).values())})
    for m in adapted(model).values():m.begin_update(False)
    if checkpoint_path is not None:
        # The retained synthetic driver runs with CUDA_VISIBLE_DEVICES='' so
        # adapter-only checkpointing cannot initialize unrelated GPU contexts.
        save_checkpoint(checkpoint_path,model,opt,{'step':stop or length,'stage_horizon':length,
            'global_update_offset':offset,'initializer_seed':seed,'score_ema':ema.state_dict(),
            'pruning_events':events,'teacher_seed':20261014,'data_seed':data['data_seed'],
            'original_backbone_sha256':hashlib.sha256(data['base'].numpy().tobytes()).hexdigest(),
            'reference_state':{key:getattr(ref,key) for key in ['ipt','exp_avg_ipt','exp_avg_unc','rank_pattern']} if ref else {}})
    return model,{'initial_dev_function_mse':initial,'history':history,'events':events,
                  'wall_s':time.perf_counter()-begin,'updates':stop or length,'score_ema_updates':ema.updates,
                  'train_examples_per_module':128,'compacted':compacted,'checkpoint':str(checkpoint_path) if checkpoint_path else None}


@torch.no_grad()
def oracle(data,budget):
    decomps={};spectra={};costs={}
    for i in range(8):
        u,s,vh=torch.linalg.svd((data['teacher'][i]@data['chol'][i]).double())
        name=f'layers.{i}';decomps[name]=(u,s,vh);spectra[name]=s;costs[name]=32
    ranks,receipt=prefix_ranks(spectra,costs,budget)
    delta=[]
    for i in range(8):
        u,s,vh=decomps[f'layers.{i}'];r=ranks[f'layers.{i}']
        delta.append(((u[:,:r]*s[:r])@vh[:r]@torch.linalg.inv(data['chol'][i].double())).float())
    prediction=data['dev']@(data['base']+torch.stack(delta)).transpose(1,2)
    return {'label':'known_teacher_covariance_weighted_rank_oracle','rank_map':ranks,'allocation':receipt,
            'dev_function_mse':float((prediction-data['dev_clean']).square().mean()),
            'dev_observed_mse':float((prediction-data['dev_noisy']).square().mean())}


def experiment(method,seed,covariance,noise,k,steps,specs,checkpoint_root=None):
    data=problem(seed,covariance,noise);spec=specs[method];stages=[];transitions=[]
    def checkpoint(name):return Path(checkpoint_root)/name if checkpoint_root else None
    if spec['group'] in {'restart','restart_control'}:
        cut=math.ceil(.2*steps);pilot_init=method.split('_PILOT_')[0] if '_PILOT_' in method else spec['init']
        pilot,rec=fit(data,specs[pilot_init+'_GATE'],k,stage_seed(seed,'pilot'),0,cut,checkpoint_path=checkpoint('pilot'));stages.append(rec)
        pattern={n:int(m.mask.sum()) for n,m in adapted(pilot).items()}
        if 'UNIFORM_RESTART' in method:pattern={n:k for n in pattern}
        elif 'SHUFFLED_SELF_RESTART' in method:pattern=shuffle_compatible(pattern,{n:(16,16) for n in pattern},stage_seed(seed,'pattern_shuffle'))
        del pilot
        model,rec=fit(data,specs[spec['init']+'_FIXED'],k,stage_seed(seed,'student'),cut,steps-cut,pattern=pattern,checkpoint_path=checkpoint('student'));stages.append(rec)
    elif spec['group']=='compress':
        cut=math.ceil(.75*steps)
        model,rec=fit(data,specs[spec['init']+'_WIDE'],k,seed,0,steps,stop=cut,checkpoint_path=checkpoint('wide_source'));stages.append(rec)
        begin=time.perf_counter()
        factors,pattern,receipt=compress_modules(adapted(model),k,1.,'global' if 'GLOBAL' in method else 'uniform')
        fresh=EightMatrices(data['base'],'L',pattern,stage_seed(seed,'recovery'))
        with torch.no_grad():
            for n,m in adapted(fresh).items():m.A.copy_(factors[n]['A']);m.B.copy_(factors[n]['B'])
        transitions.append({'compression':receipt,'wall_s':time.perf_counter()-begin,
                            'immediate_dev_function_mse':float((fresh(data['dev'])-data['dev_clean']).square().mean().detach())})
        del model
        model,rec=fit(data,specs['L_FIXED'],k,stage_seed(seed,'recovery'),cut,steps-cut,model=fresh,checkpoint_path=checkpoint('recovery'));stages.append(rec)
    else:model,rec=fit(data,spec,k,seed,0,steps,checkpoint_path=checkpoint('native'));stages.append(rec)
    assert sum(s['updates'] for s in stages)==steps
    spectra={};ranks={};allocated=active=frozen=0
    with torch.no_grad():
        for n,m in adapted(model).items():
            assert torch.equal(m.base.weight,data['base'][int(n.split('.')[-1])])
            c,d=actual_factors(m);s=torch.linalg.svdvals(c@d)
            spectra[n]=s.tolist();ranks[n]=int(m.mask.sum())
            active+=int(m.mask.sum())*m.slot_cost
            allocated+=sum(p.numel() for p in m.parameters(recurse=False))
            frozen+=sum(b.numel()*b.element_size() for b in m.buffers(recurse=False))
        pred=model(data['dev'])
    return {'method':method,'seed':seed,'covariance':covariance,'noise':noise,'steps':steps,
            'data_sha256':hashlib.sha256(b''.join(data[n].numpy().tobytes() for n in
                ['base','teacher','chol','train','targets','dev','dev_clean','dev_noisy'])).hexdigest(),
            'budget_slots_equivalent':8*k,'P_target':8*k*32,'active_scalars':active,'allocated_scalars':allocated,
            'frozen_adapter_bytes':frozen,'rank_map':ranks,'actual_update_spectra':spectra,
            'dev_function_mse':float((pred-data['dev_clean']).square().mean()),
            'dev_observed_mse':float((pred-data['dev_noisy']).square().mean()),
            'stages':stages,'transitions':transitions,'oracle':oracle(data,8*k*32),
            'teacher_ranks':TEACHER_RANKS,'teacher_seed':20261014,'scale':1.,'peak_lr':.01,
            'backbone_unchanged':True,'selection_eligible':False}


def worker(job):
    method,seed,covariance,noise,root,steps,retain=job;torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(seed);specs={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    rid=f'A_syn_{method}_{seed}_{covariance}_{noise}';out=Path(root)/rid
    if (out/'metadata.json').exists():return {'run_id':rid,'retained':True}
    out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    config={'method':method,'seed':seed,'covariance':covariance,'noise':noise,'steps':steps,'budgets':[16,32,64],
            'peak_lr':.01,'scale':1.,'teacher_seed':20261014,'training_examples_per_module':128,'dev_examples_per_module':512,
            'retain_checkpoints':retain}
    meta={'status':'running','method_id':method,'seed':seed,'config':config,'torch':torch.__version__,
          'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
          'source_revision_at_start':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*Path('src/rankstudy/v3').rglob('*.py')]}}
    json_write(out/'metadata.json',meta)
    try:
        results=[experiment(method,seed,covariance,noise,k,steps,specs,out/f'budget_{8*k}' if retain else None) for k in [2,4,8]]
        json_write(out/'results.json',results);meta.update(status='complete',cpu_wall_s=time.perf_counter()-begin,training_fits=len(results))
    except Exception as exc:meta.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}',cpu_wall_s=time.perf_counter()-begin)
    json_write(out/'metadata.json',meta)
    return {'run_id':rid,'status':meta['status'],'reason':meta.get('failure_reason')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--methods',nargs='+');ap.add_argument('--seeds',nargs='+',type=int,default=[901,902]);ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--retain-checkpoints',action='store_true')
    a=ap.parse_args();specs=yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']
    methods=a.methods or [m['id'] for m in specs]
    if a.retain_checkpoints and torch.cuda.is_available():raise ValueError("run retained CPU controls with CUDA_VISIBLE_DEVICES='' to avoid creating GPU RNG contexts")
    jobs=[(m,s,c,n,str(a.root),a.steps,a.retain_checkpoints) for s in a.seeds for c in ['isotropic','anisotropic'] for n in [False,True] for m in methods]
    a.root.mkdir(parents=True,exist_ok=True)
    json_write(a.root/'plan.json',{'jobs':[list(j[:4]) for j in jobs],'steps':a.steps,'budgets':[16,32,64],
                                  'retain_checkpoints':a.retain_checkpoints,'purpose':'common-configuration synthetic controls; no selection'})
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(worker,jobs):print(json.dumps(result),flush=True)


if __name__=='__main__':main()
