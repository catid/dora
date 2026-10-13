"""Primary-source factor recipes for the isolated DoRA initialization study.

This module returns factors only. Compensation, magnitude, and trainability are
runner decisions. See initialization_literature.md for source and adaptation
boundaries. Every decomposition uses CPU FP32 and deterministic dense solvers.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

METHODS = frozenset({'pissa', 'milora', 'olora', 'eva', 'corda', 'astra',
                     'loraga', 'loraone', 'filet', 'lorada', 'cntlora',
                     'inilora', 'tlora', 'mica'})
WEIGHT_ONLY = frozenset({'pissa', 'milora', 'olora', 'inilora', 'mica'})
_VERIFIED_CALIBRATION_FILES = {}


@contextmanager
def _prefit_precision(device):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _verify_calibration_file(path, expected):
    if expected is None:
        raise ValueError(f'Calibration checksum missing for {path}')
    path = str(Path(path).resolve())
    stat = Path(path).stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
    if _VERIFIED_CALIBRATION_FILES.get(path) != signature:
        if _sha_file(path) != expected:
            raise ValueError(f'Calibration checksum mismatch for {path}')
        _VERIFIED_CALIBRATION_FILES[path] = signature


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _tensor_hash(value):
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _statistic_binding(value):
    if isinstance(value, torch.Tensor):
        return {'tensor_sha256': _tensor_hash(value)}
    if isinstance(value, Mapping):
        return {key: _statistic_binding(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_statistic_binding(item) for item in value]
    return value


def _matrix(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must be a tensor with shape {tuple(shape)}')
    x = value.detach().to(device='cpu', dtype=torch.float32).clone()
    if not torch.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite values')
    return x


def _positive(spec, key, default):
    value = float(spec.get(key, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive')
    return value


def _canonical_columns(x):
    idx = x.abs().argmax(dim=0)
    sign = x[idx, torch.arange(x.shape[1])].sign()
    return x * torch.where(sign == 0, torch.ones_like(sign), sign)


def _cached_decomposition(x, kind, cache_dir, events):
    cache = None
    bindings = {'matrix_sha256': _tensor_hash(x), 'shape': list(x.shape),
                'dtype': str(x.dtype), 'solver': kind, 'torch_version': str(torch.__version__),
                'cpu_num_threads': torch.get_num_threads(),
                'source_sha256': _sha_file(__file__)}
    if cache_dir:
        cache = Path(cache_dir) / (_json_hash(bindings) + '.pt')
        if cache.exists():
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            if saved['bindings'] != bindings or saved['tensor_sha256'] != [_tensor_hash(v) for v in saved['values']]:
                raise ValueError('Decomposition cache integrity mismatch')
            if events is not None:
                events.append({'kind': kind, 'hit': True, 'source_seconds': saved['seconds']})
            return saved['values']
    started = time.perf_counter()
    values = (torch.linalg.svd(x, full_matrices=False) if kind == 'svd'
              else torch.linalg.eigh((x + x.T) * .5))
    elapsed = time.perf_counter() - started
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(f'.{time.time_ns()}.tmp')
        torch.save({'bindings': bindings, 'values': tuple(values),
                    'tensor_sha256': [_tensor_hash(v) for v in values], 'seconds': elapsed}, temporary)
        temporary.replace(cache)
    if events is not None:
        events.append({'kind': kind, 'hit': False, 'seconds': elapsed})
    return values


def _svd(x, cache_dir=None, events=None):
    u, s, vh = _cached_decomposition(x, 'svd', cache_dir, events)
    idx = u.abs().argmax(dim=0)
    sign = u[idx, torch.arange(u.shape[1])].sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return u * sign, s, vh * sign[:, None]


def _eigh(x, cache_dir=None, events=None):
    val, vec = _cached_decomposition(x, 'eigh', cache_dir, events)
    return val, _canonical_columns(vec)


def _spectral_factors(u, val, vh, rank, scale, spec, tail=False):
    selected = slice(-rank, None) if tail else slice(0, rank)
    p = float(spec.get('factor_power', .5))
    if not 0 <= p <= 1:
        raise ValueError('factor_power must be in [0,1]')
    values = val[selected].clamp_min(0) / scale
    return (u[:, selected] * values.pow(p),
            values.pow(1-p)[:, None] * vh[selected, :])


def _cov(calibration, key, size):
    if calibration is None or key not in calibration:
        raise ValueError(f'Initializer requires calibration statistic {key}')
    x = _matrix(calibration[key], (size, size), key)
    if not torch.allclose(x, x.T, atol=1e-5, rtol=1e-4):
        raise ValueError(f'{key} is not symmetric')
    return (x + x.T) * .5


def _selection(spec, default='top'):
    value = spec.get('selection', default)
    if value not in ('top', 'bottom'):
        raise ValueError('selection must be top or bottom')
    return value


@torch.no_grad()
def initialize_factors(weight, rank, spec, calibration=None):
    """Return (up, down, JSON metadata), without changing any input or global RNG.

    ``spec['method']`` selects a recipe. ``gain`` multiplies both factors;
    ``balance_gain`` multiplies up and divides down, preserving their product.
    ``adapter_scale`` is the source multiplier s (set 1 for accepted local DoRA).
    Source warm-start methods are merely factored here; callers must label a
    centered-compensation use as an adaptation.
    """
    started = time.perf_counter()
    if not isinstance(spec, Mapping):
        raise TypeError('spec must be a mapping')
    spec = json.loads(json.dumps(dict(spec), allow_nan=False))
    method = spec.get('method')
    if method not in METHODS:
        raise ValueError(f'Unknown data initializer {method!r}')
    if weight.ndim != 2 or not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError('Expected matrix weight and integer rank')
    out_dim, in_dim = weight.shape
    if not 1 <= rank <= min(out_dim, in_dim):
        raise ValueError('rank must be within both matrix dimensions')
    w = _matrix(weight, weight.shape, 'weight')
    scale = _positive(spec, 'adapter_scale', 1)
    gain = _positive(spec, 'gain', 1)
    balance = _positive(spec, 'balance_gain', 1)
    generator = torch.Generator(device='cpu').manual_seed(int(spec.get('seed', 42)))
    meta = {'method': method, 'spec': spec, 'weight_sha256': _tensor_hash(weight),
            'shape': [out_dim, in_dim], 'rank': rank, 'precision': 'cpu_float32', 'cpu_num_threads': torch.get_num_threads(),
            'solver': 'dense_svd_eigh_qr_sign_canonicalized',
            'source_initial_product': 'compensated', 'adaptations': [],
            'adapter_scale': scale, 'gain': gain, 'balance_gain': balance}
    events = []
    cache_dir = spec.get('decomposition_cache_dir')
    svd = lambda value: _svd(value, cache_dir, events)
    eigh = lambda value: _eigh(value, cache_dir, events)
    if method in {'pissa', 'milora'}:
        u, val, vh = svd(w)
        up, down = _spectral_factors(u, val, vh, rank, scale, spec, method == 'milora')
        selected = val[-rank:] if method == 'milora' else val[:rank]
        meta['selected_singular_values'] = selected.tolist()
    elif method == 'olora':
        q, r = torch.linalg.qr(w, mode='reduced')
        up, down = q[:, :rank], r[:rank, :]
    elif method == 'mica':
        u, _, _ = svd(w)
        up, down = u[:, -rank:], torch.zeros(rank, in_dim)
        meta['source_initial_product'] = 'zero'
        meta['adaptations'].append('MiCA initialization only; source freezes up factor')
    elif method == 'eva':
        centered = bool(spec.get('centered', True))
        key = 'input_centered_cov' if centered else 'input_cov'
        if centered and calibration is not None and key not in calibration:
            cov = _cov(calibration, 'input_cov', in_dim)
            mean = _matrix(calibration.get('input_mean'), (in_dim,), 'input_mean')
            cov -= torch.outer(mean, mean)
        else:
            cov = _cov(calibration, key, in_dim)
        val, vec = eigh(cov)
        down = vec[:, -rank:].flip(1).T
        if spec.get('whiten', False):
            count = int(calibration.get('input_count', 0))
            if count < 2:
                raise ValueError('EVA whitening requires input_count >= 2')
            # Covariance is E[(x-mean)(x-mean).T]. Singular value of the
            # activation matrix is sqrt(count*eigenvalue); PEFT then sqrt again.
            denom = (val[-rank:].flip(0).clamp_min(0) * count).pow(.25)
            floor = _positive(spec, 'whiten_floor', 1e-6)
            down = down / denom.clamp_min(floor)[:, None]
            meta['whiten_denominator'] = 'sqrt(activation_singular_value)'
        up = torch.zeros(out_dim, rank)
        meta['source_initial_product'] = 'zero'
        meta['adaptations'].append('fixed rank rho=1; dense covariance eigensolver replaces incremental PCA')
        if not centered:
            meta['adaptations'].append('uncentered activation moment')
    elif method == 'corda':
        covariance_key = spec.get('covariance_key', 'corda_input_cov' if calibration is not None and 'corda_input_cov' in calibration else 'input_cov')
        cov = _cov(calibration, covariance_key, in_dim)
        meta['input_covariance_key'] = covariance_key
        if covariance_key != 'corda_input_cov':
            meta['adaptations'].append('pooled input moment replaces source per-forward max-normalized input Gram mean')
        damping = _positive(spec, 'damping', .01)
        mean_diag = float(cov.diag().mean())
        if mean_diag <= 0:
            raise ValueError('CorDA requires positive mean activation second moment')
        eye = torch.eye(in_dim)
        adapt_damping = bool(spec.get('adaptive_damping', True))
        for attempt in range(32):
            damped = cov + damping * mean_diag * eye
            inv, info = torch.linalg.inv_ex(damped)
            residual = float(torch.linalg.matrix_norm(damped @ inv - eye))
            if int(info) == 0 and math.isfinite(residual) and residual < .05:
                break
            if not adapt_damping:
                raise ValueError(f'CorDA inverse residual {residual} exceeds .05')
            damping *= 2
        else:
            raise ValueError('CorDA inverse failed after 32 damping attempts')
        u, val, vh = svd(w @ damped)
        up, down = _spectral_factors(u, val, vh @ inv, rank, scale, spec,
                                     _selection(spec) == 'bottom')
        meta.update(damping_used=damping, inverse_residual=residual)
        if _selection(spec) == 'bottom':
            meta['adaptations'].append('task-calibrated KPM selection; original KPM uses knowledge data')
    elif method == 'astra':
        key = spec.get('covariance_key', 'astra_output_cov')
        cov = _cov(calibration, key, out_dim)
        val, vec = eigh(cov)
        up = vec[:, :rank] if _selection(spec, 'bottom') == 'bottom' else vec[:, -rank:].flip(1)
        down = up.T @ w
        if key != 'astra_output_cov':
            meta['adaptations'].append(f'Astra output covariance protocol changed to {key}')
        meta['selected_output_eigenvalues'] = (val[:rank] if _selection(spec, 'bottom') == 'bottom' else val[-rank:].flip(0)).tolist()
    elif method in {'loraga', 'loraone'}:
        if calibration is None:
            raise ValueError(f'{method} requires labeled weight_grad')
        gradient = _matrix(calibration.get('weight_grad'), w.shape, 'weight_grad')
        u, val, vh = svd(gradient)
        if method == 'loraone':
            u = -u
            if float(val[0]) <= 0:
                raise ValueError('LoRA-One requires a nonzero full-weight gradient')
            root = (val[:rank] / val[0]).sqrt()
            gamma = _positive(spec, 'gamma', 128)
            up, down = u[:, :rank] * root / math.sqrt(gamma), root[:, None] * vh[:rank] / math.sqrt(gamma)
            meta['source_initial_product'] = 'intentional_warm_start'
            meta['adaptations'].append('centered use removes the source initial gradient step; magnitude policy external')
        else:
            if 2 * rank > min(w.shape):
                raise ValueError('Author LoRA-GA disjoint subspaces require 2*rank <= min(shape)')
            direction = spec.get('direction', 'ArB2r')
            if direction == 'ArB2r':
                ai, bi = torch.arange(rank), torch.arange(rank, 2*rank)
            elif direction == 'A2rBr':
                ai, bi = torch.arange(rank, 2*rank), torch.arange(rank)
            elif direction == 'ArBr':
                ai, bi = torch.arange(1, 2*rank, 2), torch.arange(0, 2*rank, 2)
            elif direction == 'random':
                indices = torch.randperm(2*rank, generator=generator)
                ai, bi = indices[:rank], indices[rank:]
            else:
                raise ValueError(f'Unknown LoRA-GA direction {direction}')
            gamma = _positive(spec, 'gamma', 16)
            c = out_dim ** .25 / math.sqrt(gamma)
            up, down = u[:, bi] * c, vh[ai] * c
            meta.update(gradient_up_indices=bi.tolist(), gradient_down_indices=ai.tolist(),
                        scaling_recipe='author_stable_both_factors_no_singular_values')
        meta['gradient_spectral_norm'] = float(val[0])
    elif method == 'filet':
        input_key = spec.get('input_covariance_key', 'filet_input_cov' if calibration is not None and 'filet_input_cov' in calibration else 'input_cov')
        output_key = spec.get('output_covariance_key', 'filet_output_grad_cov' if calibration is not None and 'filet_output_grad_cov' in calibration else 'output_grad_cov')
        sx = _cov(calibration, input_key, in_dim)
        sy = _cov(calibration, output_key, out_dim)
        meta.update(input_covariance_key=input_key, output_covariance_key=output_key)
        if input_key != 'filet_input_cov' or output_key != 'filet_output_grad_cov':
            meta['adaptations'].append('pooled masked moments replace source unmasked mean of batch moments')
        v = F.normalize(w.T @ w, dim=0)
        u = F.normalize(w @ v, dim=0)
        score = (v * (sx @ v)).sum(0) * (u * (sy @ u)).sum(0)
        idx = torch.argsort(score, stable=True)[:rank]
        values = score[idx].clamp_min(_positive(spec, 'score_floor', 1e-8)).sqrt()
        up = u[:, idx] * values / math.sqrt(scale)
        down = values[:, None] * v[:, idx].T / math.sqrt(scale)
        meta.update(selected_indices=idx.tolist(), selected_fisher_scores=score[idx].tolist(),
                    minimum_fisher_score=float(score.min()))
    elif method == 'lorada':
        sx = _cov(calibration, 'input_cov', in_dim)
        gradient = _matrix(calibration.get('weight_grad'), w.shape, 'weight_grad')
        dy = _matrix(calibration.get('output_fisher_diag'), (out_dim,), 'output_fisher_diag')
        if (dy < 0).any():
            raise ValueError('output_fisher_diag cannot be negative')
        damping = _positive(spec, 'damping', 1e-6)
        sx += torch.eye(in_dim) * damping
        dy = (dy + damping).reciprocal()
        inv = torch.linalg.inv(sx)
        delta = -torch.linalg.solve(sx, gradient.T) * dy[None, :]
        norm = 1.
        if spec.get('normalize_delta', True):
            power = float(spec.get('delta_normalization_power', 1))
            ref = _positive(spec, 'delta_normalization_reference', 1)
            sigma = float(torch.linalg.svdvals(delta)[0])
            if sigma <= 0:
                raise ValueError('LoRA-DA requires a nonzero natural-gradient estimate')
            norm = sigma ** power * ref ** (1-power)
            delta /= norm
        population = int(spec.get('population_size', calibration.get('training_population_size', 0)))
        if population <= 0:
            raise ValueError('LoRA-DA requires positive full training population_size; calibration count is not N')
        alpha, beta = float(spec.get('bias_alpha', 1)), float(spec.get('variance_beta', .03))
        if not math.isfinite(alpha) or not math.isfinite(beta) or alpha < 0 or beta < 0:
            raise ValueError('LoRA-DA bias_alpha and variance_beta must be nonnegative')
        coefficient = float(dy.sum()) / population / norm**2
        h = -alpha * (delta @ delta.T) + beta * coefficient * inv
        val, vec = eigh(h)
        a = vec[:, :rank]
        gamma = _positive(spec, 'gamma', 1024)
        up, down = delta.T @ a / math.sqrt(gamma), a.T / math.sqrt(gamma)
        if spec.get('balance_factors', False):
            ub, sb, vhb = svd(up)
            up, down = ub * sb.sqrt(), (sb.sqrt()[:, None] * vhb) @ down
        meta.update(source_initial_product='intentional_warm_start',
                    guidance_eigenvalues=val[:rank].tolist(), variance_coefficient=coefficient,
                    delta_normalization=norm, population_size=population)
        meta['adaptations'].extend(['dense eigen/SVD solvers replace author LOBPCG/randomized SVD',
                                    'DoRA integration; source runner supports standard LoRA',
                                    'centered use removes source natural-gradient warm start'])
    elif method == 'cntlora':
        mode = spec.get('mode', 'cross')
        if mode not in {'cross', 'self', 'shift'}:
            raise ValueError('CNTLoRA mode must be cross, self, or shift')
        pooled = bool(spec.get('pooled_covariance', False))
        if pooled:
            covs = [_cov(calibration, 'input_cov', in_dim)]
            meta['adaptations'].append('pooled mean moment replaces raw per-batch input Grams')
        elif calibration is not None and 'input_gram_batches' in calibration:
            covs = calibration['input_gram_batches']
            if not len(covs):
                raise ValueError('CNTLoRA requires at least one input Gram')
        else:
            raise ValueError('CNTLoRA needs input_gram_batches or explicit pooled_covariance=True')
        rtol = _positive(spec, 'pinv_rtol', 1e-6)
        damping = float(spec.get('damping', 0))
        if not math.isfinite(damping) or damping < 0:
            raise ValueError('CNTLoRA damping must be finite and nonnegative')
        # The expensive target is independent of rank, seed, gain and factor
        # balance. Retain the exact per-batch pinv formula and accumulation order.
        target_bindings = {'weight_sha256': _tensor_hash(w),
                           'input_grams': _statistic_binding(covs),
                           'mode': mode, 'damping': damping, 'pinv_rtol': rtol,
                           'dtype': 'float32', 'device': 'cpu',
                           'torch_version': str(torch.__version__),
                           'cpu_num_threads': torch.get_num_threads(),
                           'source_sha256': _sha_file(__file__)}
        target_root = spec.get('cnt_target_cache_dir') or (str(Path(cache_dir)/'cntlora_target') if cache_dir else None)
        target_cache = None if target_root is None else Path(target_root)/(_json_hash(target_bindings)+'.pt')
        target_hit = target_cache is not None and target_cache.exists()
        if target_hit:
            saved = torch.load(target_cache, map_location='cpu', weights_only=True)
            if saved['bindings'] != target_bindings or saved['tensor_sha256'] != _tensor_hash(saved['estimated']):
                raise ValueError('CNTLoRA target cache integrity mismatch')
            estimated, target_seconds = saved['estimated'], saved['seconds']
        else:
            target_started = time.perf_counter()
            total = w.clone()
            for raw in covs:
                cov = _matrix(raw, (in_dim, in_dim), 'input_gram_batch')
                cov = (cov + cov.T) * .5 + damping * torch.eye(in_dim)
                if mode == 'cross':
                    target = w @ torch.linalg.pinv(cov, rtol=rtol, hermitian=True)
                elif mode == 'self':
                    val, vec = eigh(cov)
                    # Match descending SVD convention; this non-symmetric factor
                    # depends on its basis ordering, unlike C^(-1/2).
                    half = vec.flip(1) * val.flip(0).clamp_min(0).sqrt()
                    target = w @ torch.linalg.pinv(half, rtol=rtol)
                else:
                    target = torch.linalg.pinv(w.T - cov @ w.T, rtol=rtol)
                total += target
            estimated = total / (len(covs) + 1)
            target_seconds = time.perf_counter() - target_started
            if target_cache is not None:
                target_cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = target_cache.with_suffix(f'.{time.time_ns()}.tmp')
                torch.save({'bindings': target_bindings, 'estimated': estimated,
                            'tensor_sha256': _tensor_hash(estimated), 'seconds': target_seconds}, temporary)
                temporary.replace(target_cache)
        meta.update(cnt_target_cache_hit=target_hit, cnt_target_source_seconds=target_seconds)
        delta_sign = spec.get('delta_sign', 'target_minus_base')
        if delta_sign not in {'target_minus_base', 'base_minus_target'}:
            raise ValueError('Unknown CNTLoRA delta_sign')
        delta = estimated - w if delta_sign == 'target_minus_base' else w - estimated
        u, val, vh = svd(delta)
        up, down = _spectral_factors(u, val, vh, rank, scale, spec)
        meta.update(source_initial_product='intentional_warm_start',
                    batch_count=len(covs), delta_sign=delta_sign)
        meta['adaptations'].extend(['fixed rank; paper offers adaptive rank',
                                    'paper prose and supplement disagree on delta sign; choice explicit',
                                    'centered use removes source domain-shift warm start'])
    elif method == 'inilora':
        steps = int(spec.get('prefit_steps', 20000))
        if steps < 0:
            raise ValueError('prefit_steps must be nonnegative')
        initial_std = _positive(spec, 'initial_std', float(w.std()))
        lr = _positive(spec, 'prefit_lr', 5e-4)
        compute_device = torch.device(spec.get('prefit_device', 'cpu'))
        if compute_device.type not in {'cpu', 'cuda'}:
            raise ValueError('IniLoRA prefit_device must be cpu or cuda')
        prefit_bindings = {'weight_sha256': _tensor_hash(w), 'rank': rank,
                           'seed': int(spec.get('seed', 42)), 'steps': steps,
                           'initial_std': initial_std, 'learning_rate': lr,
                           'device': str(compute_device), 'dtype': 'float32',
                           'allow_tf32': False if compute_device.type == 'cuda' else None,
                           'torch_version': str(torch.__version__), 'cpu_num_threads': torch.get_num_threads(),
                           'source_sha256': _sha_file(__file__)}
        prefit_root = spec.get('prefit_cache_dir') or (str(Path(cache_dir)/'inilora_prefit') if cache_dir else None)
        prefit_cache = None if prefit_root is None else Path(prefit_root)/(_json_hash(prefit_bindings)+'.pt')
        prefit_hit = prefit_cache is not None and prefit_cache.exists()
        if prefit_hit:
            saved = torch.load(prefit_cache, map_location='cpu', weights_only=True)
            if saved['bindings'] != prefit_bindings or saved['tensor_sha256'] != [_tensor_hash(saved['up']), _tensor_hash(saved['down'])]:
                raise ValueError('IniLoRA prefit cache integrity mismatch')
            up, down = saved['up'], saved['down']
            initial_loss, final_loss, prefit_seconds = saved['initial_mse'], saved['final_mse'], saved['seconds']
        else:
            prefit_started = time.perf_counter()
            target = w.to(device=compute_device)
            with _prefit_precision(compute_device), torch.enable_grad(), torch.autocast(compute_device.type, enabled=False):
                up = (torch.randn(out_dim, rank, generator=generator) * initial_std).to(compute_device).requires_grad_()
                # Source fits U[out,r] @ P[in,r].T; preserve its draw layout.
                down = (torch.randn(in_dim, rank, generator=generator) * initial_std).T.contiguous().to(compute_device).requires_grad_()
                optimizer = torch.optim.Adam([up, down], lr=lr, foreach=False)
                initial_loss = float((up @ down - target).square().mean().detach())
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (up @ down - target).square().mean()
                    loss.backward()
                    # Author calls StepLR before optimizer.step, reducing on update5000.
                    optimizer.param_groups[0]['lr'] = lr * .9 ** ((step + 1) // 5000)
                    optimizer.step()
                final_loss = float((up @ down - target).square().mean().detach())
            up, down = up.detach().cpu(), down.detach().cpu()
            prefit_seconds = time.perf_counter()-prefit_started
            if prefit_cache is not None:
                prefit_cache.parent.mkdir(parents=True,exist_ok=True)
                temp = prefit_cache.with_suffix(f'.{time.time_ns()}.tmp')
                torch.save({'bindings':prefit_bindings,'up':up,'down':down,
                            'tensor_sha256':[_tensor_hash(up),_tensor_hash(down)],
                            'initial_mse':initial_loss,'final_mse':final_loss,'seconds':prefit_seconds},temp)
                temp.replace(prefit_cache)
        meta.update(prefit_steps=steps, initial_mse=initial_loss, final_mse=final_loss,
                    prefit_device=str(compute_device), prefit_compute_dtype='float32',
                    prefit_allow_tf32=prefit_bindings['allow_tf32'],
                    prefit_cache_hit=prefit_hit, prefit_source_seconds=prefit_seconds,
                    initial_std=initial_std, initial_std_policy=spec.get('initial_std_policy', 'explicit' if 'initial_std' in spec else 'per_layer'))
        if 'initial_std' not in spec:
            meta['adaptations'].append('per-layer weight std replaces rounded projection-group mean std')
        if steps != 20000:
            meta['adaptations'].append('prefit iteration budget differs from 20000-step source experiment')
    elif method == 'tlora':
        covariance_key = spec.get('covariance_key', 'tlora_input_cov' if calibration is not None and 'tlora_input_cov' in calibration else 'input_cov')
        cov = _cov(calibration, covariance_key, in_dim)
        meta['input_covariance_key'] = covariance_key
        if covariance_key != 'tlora_input_cov':
            meta['adaptations'].append('pooled input moment replaces source per-forward centered Gram mean')
        _, _, vh = svd(w @ cov)
        up, down = torch.zeros(out_dim, rank), vh[:rank, :]
        meta['source_initial_product'] = 'zero'
        meta['adaptations'].append('fixed-rank train-both TLoRA basis; source freezes down and reallocates rank/scales')
    up, down = (up * gain * balance).contiguous(), (down * gain / balance).contiguous()
    if not torch.isfinite(up).all() or not torch.isfinite(down).all():
        raise ValueError(f'{method} produced nonfinite factors')
    meta.update(up_sha256=_tensor_hash(up), down_sha256=_tensor_hash(down),
                up_rms=float(up.square().mean().sqrt()), down_rms=float(down.square().mean().sqrt()),
                up_norm=float(up.norm()), down_norm=float(down.norm()),
                elapsed_seconds=time.perf_counter()-started, decomposition_events=events)
    json.dumps(meta, allow_nan=False)
    return up, down, meta


def backend(config, factor_seed=42, calibration=None):
    """Bind recipes to the accepted runtime with per-module lazy calibration.

    Calibration supports ``module_files`` + ``module_sha256`` + JSON ``metadata``
    or an in-memory ``modules`` mapping. File checksums are mandatory for lazy
    loading. Factors can be cached under metadata.factor_cache_dir; cache keys
    bind code, spec, module, pretrained weight, seed, and calibration hashes.
    """
    from experiments import init_ablations as ab
    config = json.loads(json.dumps(dict(config), allow_nan=False))
    if config.get('method') not in METHODS:
        raise ValueError(f'Unknown initializer method {config.get("method")}')
    if float(config.get('adapter_scale', 1)) != 1:
        raise ValueError('Accepted runtime implicit adapter_scale is 1')
    code_sha = _sha_file(__file__)
    provenance = {} if calibration is None else calibration.get('metadata', {})
    calibration_binding = {'metadata': provenance,
                           'module_sha256': {} if calibration is None else calibration.get('module_sha256', {})}
    if calibration is not None and 'modules' in calibration:
        calibration_binding['in_memory_statistics_sha256'] = {
            path: _json_hash(_statistic_binding(stats))
            for path, stats in calibration['modules'].items()}
    data_spec = {'config': config, 'source_sha256': code_sha,
                 'calibration_sha256': _json_hash(calibration_binding),
                 'compensation': 'W+(BA-B0A0), same ambient autocast for both products',
                 'magnitude': 'original pretrained weight magnitude, unchanged'}

    def callback(weight, rank, module_path):
        stats, stat_sha, statistic_path = None, None, None
        if config['method'] not in WEIGHT_ONLY and calibration is not None:
            if 'module_files' in calibration and module_path in calibration['module_files']:
                statistic_path = calibration['module_files'][module_path]
                stat_sha = calibration.get('module_sha256', {}).get(module_path)
                _verify_calibration_file(statistic_path, stat_sha)
            elif 'modules' in calibration:
                stats = calibration['modules'].get(module_path)
                stat_sha = calibration_binding['in_memory_statistics_sha256'].get(module_path)
        spec = dict(config)
        if spec['method'] == 'inilora' and 'initial_std' not in spec:
            group = module_path.rsplit('.', 1)[-1]
            group_std = provenance.get('inilora_group_std', {}).get(group)
            if group_std is not None:
                spec['initial_std'] = float(group_std)
                spec['initial_std_policy'] = 'projection_group_mean_weight_std_rounded_3_decimals'
            elif spec.get('initial_std_policy') == 'group_mean_rounded':
                raise ValueError(f'IniLoRA exact group std missing for {group}')
        if provenance.get('decomposition_cache_dir'):
            spec['decomposition_cache_dir'] = provenance['decomposition_cache_dir']
        spec['seed'] = int.from_bytes(hashlib.sha256(f'{factor_seed}:{module_path}'.encode()).digest()[:8], 'little') % (2**63-1)
        key_data = dict(data_spec, factor_seed=factor_seed, cpu_num_threads=torch.get_num_threads(), module_path=module_path, rank=rank,
                        weight_sha256=_tensor_hash(weight), statistic_sha256=stat_sha)
        cache_root = provenance.get('factor_cache_dir')
        cache = None if not cache_root else Path(cache_root) / (_json_hash(key_data) + '.pt')
        if cache is not None and cache.exists():
            result = torch.load(cache, map_location='cpu', weights_only=True)
            if result['bindings'] != key_data:
                raise ValueError('Factor cache binding mismatch')
            up, down, meta = result['up'], result['down'], result['metadata']
            if _tensor_hash(up) != meta['up_sha256'] or _tensor_hash(down) != meta['down_sha256']:
                raise ValueError('Factor cache tensor checksum mismatch')
            meta = dict(meta, factor_cache_hit=True)
        else:
            if statistic_path is not None:
                # Raw per-batch Grams can be much larger than the statistics
                # this recipe needs. Map tensor storage; touch only used keys.
                stats = torch.load(statistic_path, map_location='cpu', weights_only=True, mmap=True)
            up, down, meta = initialize_factors(weight, rank, spec, stats)
            meta.update(module_path=module_path, initializer_source_sha256=code_sha,
                        calibration_sha256=stat_sha, factor_cache_hit=False)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(f'.{time.time_ns()}.tmp')
                torch.save({'bindings': key_data, 'up': up, 'down': down, 'metadata': meta}, temporary)
                temporary.replace(cache)
        return up, down, meta

    result = ab.backend(ab.InitSpec(), factor_seed=factor_seed,
                        factor_initializer=callback, data_spec=data_spec)
    result.factor_initializer = callback
    result.data_spec = data_spec
    return result
