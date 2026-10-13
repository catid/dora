"""Separate source-reviewed LoRA-Null and GME initialization extension.

This module is absent from the frozen main registry. It does not implement a
DoRA forward. Callers own model state, precision, trainability, and calibration
provenance; see initialization_extension_review.md for source divergences.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping

import torch

GME_RECIPES = ('paper', 'source_v1', 'source_v2')


def extension_families():
    """Finite pre-validation settings; a separate runner must freeze its registry."""
    families=[{'family_id':'lora_null_task_tail','parent_family':'lora_null','kind':'extension_centered',
        'source_default_index':2,'calibration_requirements':['corda_input_cov'],
        'configs':[{'method':'lora_null','calibration_role':'task_only','balance_gain':balance,
                    'covariance_key':'corda_input_cov','adapter_scale':1.}
                   for balance in (.25,.5,1.,2.)]}]
    for recipe in GME_RECIPES:
        families.append({'family_id':f'gme_{recipe}','parent_family':'gme','kind':'extension_warmstart',
            'source_default_index':1,'calibration_requirements':['coordinated_transformer_block_moments'],
            'configs':[{'method':'gme','recipe':recipe,'tau_b':tau,'probe_sigma':.02,'probe_gamma':.2,
                        'sigma_min':.005,'sigma_max':.05,'gamma_max':.8,'gamma_min':.02,
                        'response_clip':10.,'det_threshold':1e-6,'condition_threshold':1e6,'damping':1e-6,
                        'refine_steps':2,'eta_v':.5,'eta_g':.5,'eps':1e-8,
                        'magnitude_policy':'original','adapter_scale':1.,
                        'moment_policy':'paper' if recipe=='paper' else 'source_stability_floor'}
                       for tau in (3e-5,1e-4,3e-4,1e-3)]})
    return families


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _matrix(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f'{name} must be a matrix')
    result = value.detach().to(device='cpu', dtype=torch.float32).clone()
    if not torch.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def _rank(rank, shape):
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(shape):
        raise ValueError('rank must be a positive integer no larger than either weight dimension')
    return rank


def _clip(x, low, high):
    return min(high, max(low, x))


def _sign_floor(x, eps):
    return math.copysign(max(abs(x), eps), x if x != 0 else 1.)


@torch.no_grad()
def initialize_lora_null(weight, rank, input_cov, *, calibration_role='task_only', balance_gain=1.):
    """Factor W projected onto the bottom-r input-moment eigenspace.

    Returned up/down use conventional B[out,r]/A[r,in]. The caller must center
    their product to preserve the accepted starting function. Knowledge claims
    require external calibration plus retention evaluation, not task_only data.
    The thin SVD of W U avoids forming the dense W U U.T projection.
    """
    started = time.perf_counter()
    w = _matrix(weight, 'weight'); rank = _rank(rank, w.shape)
    covariance = _matrix(input_cov, 'input_cov')
    if covariance.shape != (w.shape[1], w.shape[1]):
        raise ValueError('input_cov shape must match the input width')
    if not torch.allclose(covariance, covariance.T, rtol=1e-4, atol=1e-5):
        raise ValueError('input_cov must be symmetric')
    if calibration_role not in ('task_only', 'external_knowledge'):
        raise ValueError('calibration_role must explicitly identify task_only or external_knowledge')
    balance = _positive(balance_gain, 'balance_gain')
    eigenvalues, basis = torch.linalg.eigh((covariance + covariance.T) * .5)
    if eigenvalues[0] < -1e-5 * max(1., float(eigenvalues.abs().max())):
        raise ValueError('input_cov must be positive semidefinite')
    tail = basis[:, :rank]
    projected_coordinates = w @ tail
    u, singular, vh = torch.linalg.svd(projected_coordinates, full_matrices=False)
    indices = u.abs().argmax(dim=0)
    signs = u[indices, torch.arange(rank, device='cpu')].sign(); signs[signs == 0] = 1
    u = u * signs; vh = vh * signs[:, None]
    roots = singular.sqrt()
    up = (u * roots) * balance
    down = (roots[:, None] * (vh @ tail.T)) / balance
    metadata = {'method':'lora_null', 'calibration_role':calibration_role,
        'interpretation':'task-tail initialization derivative' if calibration_role == 'task_only'
                         else 'knowledge-calibrated initialization; retention claim requires evaluation',
        'solver':'CPU FP32 symmetric eigensolver and thin projected SVD',
        'source_recipe':'LoRA-Null train-both v1, UV square-root singular split',
        'compensation_required':True, 'balance_gain':balance,
        'tail_eigenvalues':eigenvalues[:rank].tolist(),
        'boundary_eigenvalue_gap':float(eigenvalues[rank]-eigenvalues[rank-1]) if rank < len(eigenvalues) else None,
        'rank_space_reconstruction_relative_error':float(
            (up@(down@tail)-projected_coordinates).norm()/projected_coordinates.norm().clamp_min(1e-30)),
        'seconds':time.perf_counter()-started}
    return up.contiguous(), down.contiguous(), metadata


class ScalarMoments:
    """Pool valid token-by-hidden entries; FP64 sums are a numerical adaptation."""
    def __init__(self):
        self.count = 0
        self.sums = torch.zeros(3, dtype=torch.float64, device='cpu')

    @torch.no_grad()
    def add(self, hidden, mask=None):
        x = hidden.detach().to(dtype=torch.float64)
        if mask is not None:
            if tuple(mask.shape) != tuple(x.shape[:-1]):
                raise ValueError('mask must match all hidden dimensions except the final channel')
            x = x[mask.to(device=x.device, dtype=torch.bool)]
        x = x.reshape(-1)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite observed hidden states')
        self.count += x.numel()
        self.sums += torch.stack((x.sum(), x.square().sum(), x.pow(3).sum())).cpu()

    def result(self, eps=1e-8, *, source_stability_floor=False):
        eps = _positive(eps, 'eps')
        if not self.count:
            raise ValueError('no valid scalar entries observed')
        mean, second, third = (self.sums/self.count).tolist()
        variance = max(second-mean*mean, 0.)
        denominator=(variance+eps)**1.5
        if source_stability_floor:denominator=max(denominator,eps)
        return {'var':variance, 'skew':(third-3*mean*second+2*mean**3)/denominator,
                'count':self.count}


def sample_gamma(shape, sigma, skew, generator, *, source_sample_matching=False,
                 gamma_min=.02, tries=5, eps=1e-8):
    """Population-centered paper draw, or source empirical matching with retries."""
    sigma = _positive(sigma, 'sigma'); gamma_min = _positive(gamma_min, 'gamma_min')
    if not math.isfinite(float(skew)):
        raise ValueError('skew must be finite')
    if not isinstance(tries, int) or tries < 1:
        raise ValueError('tries must be a positive integer')
    if source_sample_matching and abs(skew) < 1e-4:
        x = torch.randn(shape, generator=generator, dtype=torch.float32, device='cpu')*sigma
        if source_sample_matching:
            x = (x-x.mean())*(sigma/x.std(unbiased=False).clamp_min(eps))
        return x
    concentration = (2/max(abs(skew), gamma_min))**2
    theta = sigma/math.sqrt(concentration)
    best = None; best_error = float('inf')
    for _ in range(tries if source_sample_matching else 1):
        x = (torch._standard_gamma(torch.full(shape, concentration, dtype=torch.float32, device='cpu'), generator=generator)
             * theta-concentration*theta) * (1 if skew >= 0 else -1)
        if source_sample_matching:
            x = x-x.mean(); x = x*(sigma/x.std(unbiased=False).clamp_min(eps))
        std = float(x.std(unbiased=False))
        actual_skew = float(((x-x.mean())/max(std, eps)).pow(3).mean())
        error = abs(std-sigma)+abs(actual_skew-skew)
        if error < best_error:
            best = x; best_error = error
    return best


def solve_gme_response(baseline, scale_probe, skew_probe, target, *, recipe='paper',
                       probe_sigma=.02, probe_gamma=.2, sigma_min=.005, sigma_max=.05,
                       gamma_max=.8, response_clip=10., det_threshold=1e-6,
                       condition_threshold=1e6, damping=1e-6, eps=1e-8):
    """Source-explicit response solve, including paper Eq20 ridge damping.

    The paper arm uses (M.T M + damping I)^-1 M.T, then a diagonal fallback
    when its solution is invalid; source V2 uses only its code determinant test.
    """
    if recipe not in GME_RECIPES: raise ValueError(f'unknown recipe {recipe}')
    for key,value in locals().copy().items():
        if key in ('probe_sigma','probe_gamma','sigma_min','sigma_max','gamma_max','response_clip',
                   'det_threshold','condition_threshold','damping','eps'):
            _positive(value,key)
    if sigma_min > sigma_max: raise ValueError('sigma_min exceeds sigma_max')
    for row in (baseline,skew_probe,target) + (() if scale_probe is None else (scale_probe,)):
        if not all(math.isfinite(float(row[k])) for k in ('var','skew')) or row['var'] < 0:
            raise ValueError('moments must be finite with nonnegative variance')
    dv=target['var']-baseline['var']; dg=target['skew']-baseline['skew']
    condition=None
    if recipe == 'source_v1':
        a=(skew_probe['var']-baseline['var'])/max(probe_sigma**2,eps)
        d=(skew_probe['skew']-baseline['skew'])/max(probe_gamma,eps)
        b=c=0.; mode='source_v1_diagonal'
        sigma2=max(dv,0.)/max(a,eps); gamma=dg/_sign_floor(d,eps)
    else:
        if scale_probe is None: raise ValueError('two-probe recipe requires scale_probe')
        denominator=probe_sigma**2+eps if recipe=='paper' else max(probe_sigma**2,eps)
        a=(scale_probe['var']-baseline['var'])/denominator
        c=(scale_probe['skew']-baseline['skew'])/denominator
        reference=scale_probe if recipe=='paper' else baseline
        denominator=probe_gamma+eps if recipe=='paper' else probe_gamma
        b=(skew_probe['var']-reference['var'])/denominator
        d=(skew_probe['skew']-reference['skew'])/denominator
        a,b,c,d=(_clip(x,-response_clip,response_clip) for x in (a,b,c,d))
        if recipe=='source_v2':
            a=_sign_floor(a,eps); d=_sign_floor(d,eps)
        determinant=a*d-b*c
        matrix=torch.tensor([[a,b],[c,d]],dtype=torch.float64,device='cpu')
        condition=float(torch.linalg.cond(matrix))
        ill=abs(determinant)<det_threshold or (recipe=='paper' and condition>condition_threshold)
        mode='coupled'
        if ill and recipe=='paper':
            mode='paper_ridge_damped'
            try:
                target_vector=torch.tensor([dv,dg],dtype=torch.float64,device='cpu')
                solution=torch.linalg.solve(matrix.T@matrix+damping*torch.eye(2,dtype=torch.float64,device='cpu'),
                                            matrix.T@target_vector)
                sigma2,gamma=solution.tolist()
            except torch.linalg.LinAlgError:
                sigma2=gamma=float('nan')
            if not (math.isfinite(sigma2) and math.isfinite(gamma) and
                    sigma_min**2 <= sigma2 <= sigma_max**2 and abs(gamma)<=gamma_max):
                mode='fallback_diagonal'
        elif ill:
            mode='fallback_diagonal'
        else:
            sigma2=(d*dv-b*dg)/determinant; gamma=(-c*dv+a*dg)/determinant
        if mode=='fallback_diagonal':
            sigma2=max(dv,0.)/max(a,eps); gamma=dg/_sign_floor(d,eps)
    sigma=_clip(math.sqrt(max(sigma2,0.)),sigma_min,sigma_max)
    gamma_clipped=_clip(gamma,-gamma_max,gamma_max)
    rho=sigma/probe_sigma
    if recipe!='paper':rho=_clip(rho,.1,10.)
    return {'a':a,'b':b,'c':c,'d':d,'determinant':a*d-b*c,'solve_mode':mode,
            'condition_number':condition if condition is not None and math.isfinite(condition) else None,
            'condition_number_nonfinite':condition is not None and not math.isfinite(condition),
            'zero_probe_response':all(skew_probe[k]==baseline[k] and
                                      (scale_probe is None or scale_probe[k]==baseline[k]) for k in ('var','skew')),
            'sigma2_raw':sigma2,'gamma_raw':gamma,'sigma':sigma,'gamma':gamma_clipped,'rho':rho}


@torch.no_grad()
def initialize_gme(weights: Mapping, ranks, module_layers: Mapping, observe, *, recipe='paper', seed=42,
                   tau_b=1e-4, gamma_min=.02, refine_steps=2, eta_v=.5, eta_g=.5,
                   **solver_options):
    """Coordinate whole-model probes through observe(factors_or_None, stage).

    factors[module]=(up,down), CPU FP32. None requests accepted zero-update DoRA.
    observe must install every factor together, keep the original DoRA magnitude,
    run the same training-only calibration in eval mode, and return a mapping of
    block ID to {'var','skew','count'}. None uses the accepted zero-update DoRA
    forward, never raw Linear. Use one call per moment domain. The caller
    must restore temporary model state on errors. This is a nonzero warm start;
    compensation would erase the response that the method aims to initialize.
    """
    started=time.perf_counter()
    if recipe not in GME_RECIPES:raise ValueError(f'unknown recipe {recipe}')
    if not weights or set(weights)!=set(module_layers):raise ValueError('weights and module_layers must match')
    if isinstance(refine_steps,bool) or not isinstance(refine_steps,int) or refine_steps<0:
        raise ValueError('refine_steps must be a nonnegative integer')
    tau_b=_positive(tau_b,'tau_b');gamma_min=_positive(gamma_min,'gamma_min')
    eta_v=_positive(eta_v,'eta_v');eta_g=_positive(eta_g,'eta_g')
    tensors={name:_matrix(w,name) for name,w in weights.items()}
    rank_by_name={name:_rank(ranks[name] if isinstance(ranks,Mapping) else ranks,w.shape)
                  for name,w in tensors.items()}
    layers=list(dict.fromkeys(module_layers.values()))
    generator=torch.Generator(device='cpu').manual_seed(seed)
    options={'probe_sigma':.02,'probe_gamma':.2,'sigma_min':.005,'sigma_max':.05,'gamma_max':.8,
             'eps':1e-8}|solver_options
    observations={};counts=None

    def measure(factors,stage):
        nonlocal counts
        raw=observe(factors,stage)
        if set(raw)!=set(layers):raise ValueError('observed layers differ from declared moment domain')
        rows={layer:{k:float(raw[layer][k]) for k in ('var','skew','count')} for layer in layers}
        for row in rows.values():
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError('Nonfinite observed moments')
            if row['var']<0 or row['count']<=0:
                raise ValueError('invalid observed variance or scalar count')
        current={layer:rows[layer]['count'] for layer in layers}
        if counts is not None and counts!=current:raise ValueError('calibration scalar counts changed between probes')
        counts=current;observations[stage]=rows
        return rows

    def sample(solutions,tries):
        factors={}
        for name,w in tensors.items():
            solution=solutions[module_layers[name]];rank=rank_by_name[name]
            # Preserve source order: draw down A, then up B.
            down=sample_gamma((rank,w.shape[1]),solution['sigma'],solution['gamma'],generator,
                              source_sample_matching=recipe!='paper',gamma_min=gamma_min,tries=tries)
            up=torch.randn((w.shape[0],rank),generator=generator,dtype=torch.float32,device='cpu')*(tau_b*solution['rho']/math.sqrt(rank))
            factors[name]=(up,down)
        return factors

    baseline=measure(None,'baseline')
    target={k:float(torch.tensor([baseline[l][k] for l in layers],dtype=torch.float32,device='cpu').median())
            for k in ('var','skew')}
    probe=lambda gamma:{l:{'sigma':options['probe_sigma'],'gamma':gamma,'rho':1.} for l in layers}
    scale_probe=None
    if recipe!='source_v1':scale_probe=measure(sample(probe(0.),4),'probe_scale')
    skew_probe=measure(sample(probe(options['probe_gamma']),4),'probe_skew')
    solved={l:solve_gme_response(baseline[l],None if scale_probe is None else scale_probe[l],
                                skew_probe[l],target,recipe=recipe,**options) for l in layers}
    initial_solution={l:dict(row) for l,row in solved.items()}
    factors=sample(solved,5)
    for index in range(refine_steps):
        current=measure(factors,f'refine_{index}')
        for layer in layers:
            solution=solved[layer]
            exponent=eta_v*(target['var']-current[layer]['var'])/max(target['var'],options['eps'])
            # Clipping the exponent prevents overflow without changing clipped sigma.
            solution['sigma']=_clip(solution['sigma']*math.exp(_clip(exponent,-700.,700.)),
                                    options['sigma_min'],options['sigma_max'])
            solution['gamma']=_clip(solution['gamma']+eta_g*(target['skew']-current[layer]['skew']),
                                    -options['gamma_max'],options['gamma_max'])
            solution['rho']=solution['sigma']/options['probe_sigma']
            if recipe!='paper':solution['rho']=_clip(solution['rho'],.1,10.)
        factors=sample(solved,5)
    measure(factors,'final')
    ratios={name:float(up.norm()/tensors[name].norm().clamp_min(options['eps']))
            for name,(up,_) in factors.items()}
    metadata={'method':'gme','recipe':recipe,'seed':seed,'target':target,'initial_solution':initial_solution,
        'final_solution':solved,'observations':observations,'solver_options':options,'tau_b':tau_b,
        'gamma_min':gamma_min,'refine_steps':refine_steps,'eta_v':eta_v,'eta_g':eta_g,
        'compensation_required':False,'initial_function':'intentional nonzero warm start',
        'magnitude_policy':'retain accepted base magnitude throughout probes and final initialization',
        'sample_policy':'source empirical mean/std matching and best-skew retries; Gaussian near-zero skew' if recipe!='paper'
                        else 'population-centered single Gamma draw; gamma_min floor even at zero skew',
        'required_moment_policy':'paper' if recipe=='paper' else 'source_stability_floor',
        'target_median':'FP32 lower median across blocks within one moment domain',
        'b_norm_ratios':ratios,'b_norm_ratio_ok':max(ratios.values())<1e-3,
        'seconds':time.perf_counter()-started}
    return factors,metadata
