"""Initializer-only experiments on accepted DoRA; public and frozen sources untouched.

All recipes consume the original initializer's global random stream first.
Controlled recipes then replace factor values with draws from a local CPU
generator, preserving downstream sampling/dropout RNG state across recipes.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments import change_ablations as core
from experiments import next_ablations as previous
from experiments import nora_current as current


@dataclass(frozen=True)
class InitSpec:
    mode: str = 'accepted'
    orientation: str = 'up'
    family: str = 'normal'
    scale: float = 1.
    normalization: str = 'analytic_rms'
    df: float = 5.
    truncation: float = 2.
    sparsity: float = .1
    kernel: str = 'none'
    kernel_length: float = 1.
    radial: str = 'none'
    both_nonzero: bool = False
    companion_scale: float = 1.
    scaling: str = 'rank'
    permute: bool = False

    def __post_init__(self):
        if self.mode not in ('accepted', 'accepted_scaled', 'controlled'):
            raise ValueError('Unknown initializer mode')
        if self.orientation not in ('up', 'down'):
            raise ValueError('orientation must be up or down')
        if self.family not in ('normal', 'uniform', 'rademacher', 'laplace', 'student_t',
                               'truncnormal', 'orthogonal', 'sparse', 'dct', 'hadamard', 'cropped_hadamard','bimi','peft_orthogonal'):
            raise ValueError('Unknown noise family')
        if self.normalization not in ('analytic_rms', 'sample_rms'):
            raise ValueError('Unknown RMS convention')
        if self.kernel not in ('none', 'rbf', 'exponential', 'matern32'):
            raise ValueError('Unknown rank-axis covariance kernel')
        if self.radial not in ('none','unit_vectors') or self.scaling not in ('rank','fan_in','original_input','absolute'):
            raise ValueError('Unknown radial/scaling policy')
        if not isinstance(self.both_nonzero,bool) or not isinstance(self.permute,bool):
            raise ValueError('both_nonzero and permute must be booleans')
        if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0
               for x in (self.scale, self.df, self.truncation, self.sparsity, self.kernel_length,self.companion_scale)):
            raise ValueError('All numeric initializer parameters must be finite and positive')
        if self.df <= 2 or self.sparsity > 1 or self.truncation < .1:
            raise ValueError('Require df>2, 0<sparsity<=1, truncation>=.1')
        if self.mode != 'controlled':
            if self.orientation != 'up' or self.family != 'normal' or self.normalization != 'analytic_rms' or self.kernel != 'none' or self.radial!='none' or self.both_nonzero or self.scaling!='rank' or self.permute:
                raise ValueError('Accepted modes preserve original orientation, family, RMS convention and independence')
            if self.mode == 'accepted' and self.scale != 1.:
                raise ValueError('Use accepted_scaled for original-draw scale changes')
        if self.family=='peft_orthogonal' and (not self.both_nonzero or self.scaling!='absolute' or self.normalization!='analytic_rms' or self.kernel!='none' or self.radial!='none'):
            raise ValueError('PEFT orthogonal requires both_nonzero, absolute scaling and native complementary-subspace construction')
        if self.permute and self.family != 'bimi':
            raise ValueError('permute is only defined for BIMI')


SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PUBLIC_SOURCE_SHA256 = previous.PUBLIC_SOURCE_SHA256


def _spec(value):
    if isinstance(value, InitSpec): return value
    if isinstance(value, dict): return InitSpec(**value)
    raise TypeError('spec must be InitSpec or its serialized mapping')


def recipe_manifest(spec, method=None, data_spec=None):
    spec = _spec(spec)
    payload = {
        'schema': 'dora-initializer-v1', 'method': method or 'initializer', 'initializer': asdict(spec),
        **asdict(previous.REGISTRY['next_base']),
        'factor_names': 'lora_A=[rank,input] down; lora_B=[output,rank] up',
        'random_factor': 'lora_B' if spec.orientation == 'up' else 'lora_A',
        'zero_initialized_factor': 'lora_A' if spec.orientation == 'up' else 'lora_B',
        'trainable_registration_order': ['m', 'lora_B', 'lora_A'],
        'control_method': 'current_dora', 'base_method': 'next_base',
        'stream_policy': 'consume_original_then_local_replace' if spec.mode == 'controlled' else 'original_global_cpu_draws',
        'local_seed_policy': 'SHA256 compact sorted JSON [factor_seed,module_path,orientation], first8bytes little-endian modulo2**63; family excluded to pair local streams',
        'factor_storage': 'Original FP32 factors drawn on CPU then device-only move; FP64 conversion remains explicit',
        'rms_convention': 'Analytic expected squared entry scale**2/rank; equals variance for zero-mean families. Deterministic structures match RMS; sample RMS may fluctuate.' if spec.normalization == 'analytic_rms' else 'Exact sample Frobenius RMS scale/sqrt(rank), applied after any rank covariance',
        'kernel_scope': 'Stationary covariance across arbitrary rank coordinates spaced one unit apart, unit diagonal; not semantic neuron geometry',
        'orthogonal_scope': 'QR/DCT/Hadamard orthonormal rank columns in [nonrank,rank], scaled sqrt(nonrank/rank); row norms not constrained. Hadamard requires power-of-two nonrank width. Cropped Hadamard takes the first nonrank rows of the next-power-of-two Sylvester basis with per-entry RMS preserved; columns need not remain orthogonal.',
        'sparse_scope': 'Independent Bernoulli(sparsity) entries times Rademacher/sqrt(sparsity); zero vectors recorded, never repaired',
        'formula': 'Accepted dense m*(W+up@down)/column_norm(W+up@down), full norm gradients, native norm arithmetic, implicit scale1',
        'compensation': 'none; exactly one factor zero, initial adapter update zero',
        'scope': 'Initializer-only family/orientation/scale/covariance intervention. No runtime NoRA; no data-aware or both-nonzero compensation mode.',
    }
    payload['scaling_policy']=spec.scaling
    if spec.scaling=='fan_in':payload['rms_convention']='Per-entry standard deviation scale/sqrt(fan_in); up fan_in=rank, down fan_in=input width; exact-sample RMS only when explicitly requested'
    elif spec.scaling=='original_input':payload['rms_convention']='Both factors use original base input width: per-entry standard deviation scale/sqrt(input), irrespective of factor orientation; uniform scale1/sqrt3 means bounds +/-1/sqrt(input)'
    elif spec.scaling=='absolute':payload['rms_convention']='Per-entry standard deviation scale, or exact sample RMS scale when explicitly requested'
    if spec.radial=='unit_vectors':payload['rms_convention']='One-time rank-vector unit L2 normalization after sampling/covariance, then multiply scale; per-entry RMS becomes scale/sqrt(rank). Zero sampled vectors rejected.'
    if spec.both_nonzero:
        payload.update(zero_initialized_factor=None,compensation='Centered adapter product using detached initial factor buffers, same ambient autocast',scope='Both-nonzero initialization with centered compensation; distinct from one-zero-factor initializer-only recipes')
    if spec.family=='bimi':payload['structured_formula']='Repeated identity: random-factor rank vector i is e_(i mod rank), multiplied scale under rank policy. Optional local random row/rank permutation and row signs explicitly declared.'
    if spec.family=='peft_orthogonal':
        payload['structured_formula']='PEFT complementary subspaces: Q=qr(randn(rank,rank)); down=(randn(in,rank/2)@Q[0::2,:]).T/10; up=randn(rank/2,out).T@Q[1::2,:]/10. Even rank required; scale and companion_scale multiply designated primary and secondary factors.'
        payload['rms_convention']='Published /10 complementary-subspace draws; no extra RMS normalization. Each factor expected entry variance .005 before explicit scale multipliers.'
    if data_spec is not None:
        payload.update(initializer_kind='data', data_initializer=json.loads(json.dumps(data_spec,allow_nan=False)),
                       compensation='Centered adapter product: W+(up@down-initial_up@initial_down); both products evaluated under identical ambient autocast',
                       stream_policy='consume_original_then_local_callback',
                       zero_initialized_factor='Determined from callback factors; may be neither',
                       rms_convention='Callback-defined factors; no random-family RMS rescaling applied',
                       factor_storage='Callback factors copied to existing FP32 factor storage; m remains native pretrained norm promoted to FP32',
                       scope='Data/weight-aware callback with explicit centered compensation; distinct from initializer-only one-zero-factor random recipes')
    else:
        payload['initializer_kind']='random'
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return {**payload, 'recipe_sha256': hashlib.sha256(encoded).hexdigest(),
            'source_sha256': SOURCE_SHA256, 'inherited_current_source_sha256': current.SOURCE_SHA256,
            'inherited_next_source_sha256': previous.SOURCE_SHA256,
            'inherited_core_source_sha256': core.SOURCE_SHA256, 'public_source_sha256': PUBLIC_SOURCE_SHA256}


def local_seed(factor_seed, module_path, orientation):
    if isinstance(factor_seed, bool) or not isinstance(factor_seed, int):
        raise ValueError('factor_seed must be an integer')
    if not isinstance(module_path, str): raise ValueError('module_path must be a string')
    encoded = json.dumps([factor_seed, module_path, orientation], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little') % (2 ** 63)


def rank_covariance(rank, spec):
    distance = (torch.arange(rank, dtype=torch.float64)[:, None] - torch.arange(rank, dtype=torch.float64)[None, :]).abs()
    distance = distance / spec.kernel_length
    if spec.kernel == 'none': return torch.eye(rank, dtype=torch.float64)
    if spec.kernel == 'rbf': return torch.exp(-.5 * distance.square())
    if spec.kernel == 'exponential': return torch.exp(-distance)
    scaled = math.sqrt(3) * distance
    return (1 + scaled) * torch.exp(-scaled)


def draw_factor(spec, shape, generator, fan_in=None):
    """Return [nonrank,rank] noise with the explicitly selected RMS convention."""
    vectors, rank = shape
    family = spec.family
    if family == 'normal': value = torch.randn(shape, generator=generator)
    elif family == 'uniform': value = (torch.rand(shape, generator=generator) * 2 - 1) * math.sqrt(3)
    elif family == 'rademacher': value = torch.randint(2, shape, generator=generator).float().mul_(2).sub_(1)
    elif family == 'laplace':
        u = torch.rand(shape, generator=generator).double().clamp_min(torch.finfo(torch.float32).eps) - .5
        value = (-u.sign() * torch.log1p(-2 * u.abs()) / math.sqrt(2)).float()
    elif family == 'student_t':
        gaussian = torch.randn(shape, generator=generator)
        chi_square = 2 * torch._standard_gamma(torch.full(shape, spec.df / 2), generator=generator)
        value = gaussian / (chi_square / (spec.df - 2)).sqrt()
    elif family == 'truncnormal':
        a = spec.truncation
        lower = .5 * (1 - math.erf(a / math.sqrt(2)))
        u = lower + (1 - 2 * lower) * torch.rand(shape, generator=generator, dtype=torch.float64)
        u = u.clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
        variance = 1 - 2 * a * math.exp(-a*a/2) / math.sqrt(2*math.pi) / (1 - 2*lower)
        value = (math.sqrt(2) * torch.erfinv(2*u - 1) / math.sqrt(variance)).float()
    elif family == 'sparse':
        mask = torch.rand(shape, generator=generator) < spec.sparsity
        signs = torch.randint(2, shape, generator=generator).float().mul_(2).sub_(1)
        value = mask * signs / math.sqrt(spec.sparsity)
    elif family=='bimi':
        value=torch.zeros(shape)
        value[torch.arange(vectors),torch.arange(vectors)%rank]=math.sqrt(rank)
        if spec.permute:
            value=value[torch.randperm(vectors,generator=generator)][:,torch.randperm(rank,generator=generator)]
            value=value*torch.randint(2,(vectors,1),generator=generator).mul_(2).sub_(1)
    elif family=='peft_orthogonal':raise ValueError('Complementary PEFT orthogonal must be drawn as a factor pair')
    else:
        if vectors < rank: raise ValueError('Structured orthogonal families require nonrank width >= rank')
        if family == 'orthogonal':
            value, triangular = torch.linalg.qr(torch.randn(shape, generator=generator, dtype=torch.float64), mode='reduced')
            signs = triangular.diag().sign(); signs[signs == 0] = 1
            value = value * signs
        else:
            basis_width = 1 << (vectors - 1).bit_length() if family == 'cropped_hadamard' else vectors
            columns = torch.randperm(basis_width, generator=generator)[:rank]
            rows = torch.arange(vectors)
            if family == 'dct':
                value = torch.cos(math.pi / vectors * (rows.double()[:, None] + .5) * columns.double()[None, :]) * math.sqrt(2 / vectors)
                value[:, columns == 0] /= math.sqrt(2)
            else:
                if family == 'hadamard' and vectors & (vectors - 1): raise ValueError('Hadamard requires power-of-two nonrank width')
                bits = rows[:, None].bitwise_and(columns[None, :])
                parity = torch.zeros_like(bits)
                for shift in range(basis_width.bit_length()): parity.bitwise_xor_((bits >> shift) & 1)
                value = (1 - 2 * parity).double() / math.sqrt(vectors)
            row_signs = torch.randint(2, (vectors, 1), generator=generator).mul_(2).sub_(1)
            column_signs = torch.randint(2, (1, rank), generator=generator).mul_(2).sub_(1)
            value = value * row_signs * column_signs
        value = (value * math.sqrt(vectors)).float()
    if spec.kernel != 'none':
        eigenvalues, eigenvectors = torch.linalg.eigh(rank_covariance(rank, spec))
        if float(eigenvalues.min()) < -1e-10: raise ValueError('Rank covariance is not positive semidefinite')
        square_root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
        value = (value.double() @ square_root.T).float()
    if spec.normalization == 'sample_rms':
        rms = value.double().square().mean().sqrt()
        if not bool(rms > 0): raise ValueError('Cannot RMS-normalize an all-zero sampled factor')
        value = (value.double() / rms).float()
    if spec.radial=='unit_vectors':
        lengths=value.norm(dim=1,keepdim=True)
        if bool((lengths==0).any()):raise ValueError('Cannot normalize zero sampled rank vectors; no silent repair')
        return value/lengths*spec.scale
    denominator=1 if spec.scaling=='absolute' else (fan_in or rank) if spec.scaling in ('fan_in','original_input') else rank
    return value * (spec.scale / math.sqrt(denominator))


class AblationLinear(previous.AblationLinear):
    def __init__(self, base, spec, rank=8, module_path='', factor_seed=42, method=None,
                 factor_initializer=None, data_spec=None):
        spec = _spec(spec)
        # Preserve original global draws, parameter order, magnitude computation,
        # dtype and device movement before any controlled local replacement.
        super().__init__(base, 'next_base', rank)
        self.init_spec, self.module_path, self.factor_seed = spec, module_path, factor_seed
        self.data_spec = None if data_spec is None else json.loads(json.dumps(data_spec,allow_nan=False))
        self.compensated = False
        self.method = method or 'initializer'
        self.zero_initialized_factor = 'lora_A' if spec.orientation == 'up' else 'lora_B'
        self._local_seed = local_seed(factor_seed, module_path, spec.orientation) if spec.mode == 'controlled' else None
        pair=None
        with torch.no_grad():
            if spec.mode == 'accepted_scaled': self.lora_B.mul_(spec.scale)
            elif spec.mode == 'controlled':
                vectors = self.out_features if spec.orientation == 'up' else self.in_features
                generator = torch.Generator(device='cpu').manual_seed(self._local_seed)
                if spec.family=='peft_orthogonal':
                    if rank%2:raise ValueError('PEFT complementary orthogonal initialization requires even rank')
                    q,_=torch.linalg.qr(torch.randn(rank,rank,generator=generator))
                    down=(torch.randn(self.in_features,rank//2,generator=generator)@q[0::2,:]).T/10
                    up=torch.randn(rank//2,self.out_features,generator=generator).T@q[1::2,:]/10
                    up*=spec.scale if spec.orientation=='up' else spec.companion_scale
                    down*=spec.scale if spec.orientation=='down' else spec.companion_scale
                    pair=(up,down)
                else:
                    primary_fan_in = self.in_features if spec.scaling=='original_input' or spec.orientation=='down' else rank
                    random = draw_factor(spec, (vectors, rank), generator,primary_fan_in)
                    self.lora_A.zero_(); self.lora_B.zero_()
                    if spec.orientation == 'up': self.lora_B.copy_(random.to(self.lora_B.device))
                    else: self.lora_A.copy_(random.T.to(self.lora_A.device))
                    if spec.both_nonzero:
                        companion=replace(spec,orientation='down' if spec.orientation=='up' else 'up',scale=spec.companion_scale)
                        other_vectors=self.in_features if spec.orientation=='up' else self.out_features
                        other_fan_in = self.in_features if spec.scaling=='original_input' or spec.orientation=='up' else rank
                        other=draw_factor(companion,(other_vectors,rank),generator,other_fan_in)
                        pair=(random,other.T) if spec.orientation=='up' else (other,random.T)
        if not torch.isfinite(self.lora_A).all() or not torch.isfinite(self.lora_B).all():
            raise ValueError('Initializer produced nonfinite factor storage')
        self._initialization_diagnostics = self._initial_diagnostics()
        if pair is not None:
            self.initialize_from_factors(*pair,{'kind':'random','spec':asdict(spec),'local_seed':self._local_seed},compensate=True)
            self._initialization_diagnostics['initializer_kind']='random_both_nonzero'
        if factor_initializer is not None:
            if data_spec is None or spec != InitSpec():
                raise ValueError('Data callback requires explicit data_spec and unmodified accepted InitSpec')
            cpu_rng=torch.get_rng_state()
            cuda_rng=torch.cuda.get_rng_state(self.weight.device) if self.weight.is_cuda else None
            up,down,metadata=factor_initializer(self.weight,rank,module_path)
            if not torch.equal(cpu_rng,torch.get_rng_state()):
                raise ValueError('Factor callback changed global CPU RNG; use a local generator')
            if cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(self.weight.device)):
                raise ValueError('Factor callback changed global CUDA RNG; use a local generator')
            self.initialize_from_factors(up,down,metadata,compensate=True)
        elif data_spec is not None:
            raise ValueError('data_spec requires a factor_initializer callback')

    def initialize_from_factors(self, up, down, metadata, compensate=True):
        if self.compensated: raise ValueError('Factors already initialized from callback')
        if tuple(up.shape)!=(self.out_features,self.rank) or tuple(down.shape)!=(self.rank,self.in_features):
            raise ValueError('Callback factor shapes do not match layer')
        if not up.is_floating_point() or not down.is_floating_point() or not torch.isfinite(up).all() or not torch.isfinite(down).all():
            raise ValueError('Callback factors must be finite floating tensors')
        if not compensate: raise ValueError('Callback modes currently require explicit centered compensation')
        metadata=json.loads(json.dumps(metadata,allow_nan=False))
        with torch.no_grad():
            self.lora_B.copy_(up.to(device=self.lora_B.device,dtype=self.lora_B.dtype))
            self.lora_A.copy_(down.to(device=self.lora_A.device,dtype=self.lora_A.dtype))
        if not torch.isfinite(self.lora_B).all() or not torch.isfinite(self.lora_A).all():
            raise ValueError('Callback factors overflowed trainable storage')
        self.register_buffer('initial_up',self.lora_B.detach().clone())
        self.register_buffer('initial_down',self.lora_A.detach().clone())
        self.compensated=True
        up_zero=bool(torch.count_nonzero(self.lora_B.detach())==0)
        down_zero=bool(torch.count_nonzero(self.lora_A.detach())==0)
        self.zero_initialized_factor='lora_A' if down_zero else ('lora_B' if up_zero else None)
        def digest(value):
            value=value.detach().cpu().contiguous()
            return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        self._initialization_diagnostics={
            'factor_seed':self.factor_seed,'module_path':self.module_path,'initializer_kind':'data',
            'compensated':True,'up_initially_zero':up_zero,'down_initially_zero':down_zero,
            'initial_adapter_delta_is_zero':True,'up_frobenius_norm':float(self.initial_up.double().norm()),
            'down_frobenius_norm':float(self.initial_down.double().norm()),
            'initial_up_sha256':digest(self.initial_up),'initial_down_sha256':digest(self.initial_down),
            'callback_metadata':metadata,
            'reload_policy':'Recreate initial factor buffers from bound callback/calibration, verify content hashes; trainable-only checkpoints omit these buffers',
            'runtime_cost':'Two factor products per forward; frozen initial product recomputed under identical autocast',
        }
        for label,factor in (('up',self.initial_up),('down',self.initial_down.T)):
            factor=factor.detach().cpu().double()
            gram=factor.T@factor
            values=torch.linalg.eigvalsh(gram).clamp_min(0)
            mass=values.sum();probabilities=values/mass if mass>0 else values
            positive=probabilities[probabilities>0]
            self._initialization_diagnostics[label+'_statistics']={
                'entry_rms':float(factor.square().mean().sqrt()),'rank_gram':gram.tolist(),
                'rank_gram_eigenvalues':values.tolist(),
                'effective_rank':float(torch.exp(-(positive*positive.log()).sum())) if mass>0 else 0.,
                'zero_rank_vectors':int((factor.norm(dim=1)==0).sum()),
                'minimum_rank_vector_norm':float(factor.norm(dim=1).min()),
                'maximum_rank_vector_norm':float(factor.norm(dim=1).max()),
            }

    def forward(self,x):
        if not self.compensated:
            return super().forward(x)
        update=self.lora_B@self.lora_A-self.initial_up@self.initial_down
        adapted=self.weight+update
        weights=self.m*(adapted/self._weight_norm(adapted))
        return core._project(x,weights,self.bias)

    def _initial_diagnostics(self):
        random = (self.lora_B if self.init_spec.orientation == 'up' else self.lora_A.T).detach().cpu().double()
        gram = random.T @ random
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        mass = eigenvalues.sum()
        probabilities = eigenvalues / mass if mass > 0 else torch.zeros_like(eigenvalues)
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(nonzero * nonzero.log()).sum())) if mass > 0 else 0.
        zero = getattr(self, self.zero_initialized_factor).detach()
        return {'factor_seed': self.factor_seed, 'module_path': self.module_path, 'local_seed': self._local_seed,
                'initializer_kind':'random',
                'random_factor_orientation': self.init_spec.orientation,
                'random_factor_rms': float(random.square().mean().sqrt()),
                'target_entry_rms': self.init_spec.scale / math.sqrt(self.rank if self.init_spec.radial=='unit_vectors' else 1 if self.init_spec.scaling=='absolute' else self.in_features if self.init_spec.scaling=='original_input' or (self.init_spec.scaling=='fan_in' and self.init_spec.orientation=='down') else self.rank),
                'random_factor_frobenius_norm': float(random.norm()),
                'random_vector_count': random.shape[0],
                'random_zero_vectors': int((random.norm(dim=1) == 0).sum()),
                'random_minimum_vector_norm': float(random.norm(dim=1).min()),
                'random_maximum_vector_norm': float(random.norm(dim=1).max()),
                'rank_gram': gram.tolist(), 'rank_gram_eigenvalues': eigenvalues.tolist(),
                'rank_effective_rank': effective_rank,
                'kernel_eigenvalues': torch.linalg.eigvalsh(rank_covariance(self.rank, self.init_spec)).tolist(),
                'zero_factor_nonzero_entries': int(torch.count_nonzero(zero)),
                'initial_adapter_delta_is_zero': bool(torch.count_nonzero(zero) == 0)}

    def norm_diagnostics(self):
        return {'normalization_axis': self.norm_axis, 'zero_initialized_factor': self.zero_initialized_factor,
                'minimum_raw_norm': float(self._minimum_raw_norm), 'clamp_activations': int(self._clamp_activations),
                'norm_evaluations': self._norm_evaluations, 'norm_floor': None,
                'magnitude_dtype': str(self.m.dtype), 'factor_dtype': str(self.lora_A.dtype),
                'recipe_sha256': recipe_manifest(self.init_spec, self.method, self.data_spec)['recipe_sha256'],
                'initializer': self._initialization_diagnostics}


def inject_adapters(model, spec, rank=8, targets=None, factor_seed=42, method=None,
                    factor_initializer=None,data_spec=None):
    spec = _spec(spec)
    if isinstance(targets, str): targets = (targets,)
    if targets is None: matches = lambda name, module: True
    elif callable(targets): matches = targets
    else:
        suffixes = tuple(targets)
        matches = lambda name, module: any(name == item or name.endswith('.' + item) for item in suffixes)
    selected = {id(module): name for name, module in reversed(list(model.named_modules(remove_duplicate=False)))
                if isinstance(module, nn.Linear) and matches(name, module)}
    if not selected: raise ValueError('no nn.Linear modules matched adapter targets')
    canonical = {id(module): name for name, module in reversed(list(model.named_modules()))}
    model.requires_grad_(False)
    replacements = {}
    def rewrite(module):
        if id(module) in replacements: return replacements[id(module)]
        result = AblationLinear(module, spec, rank, canonical[id(module)], factor_seed, method,
                                factor_initializer,data_spec) if id(module) in selected else module
        replacements[id(module)] = result
        if result is module:
            for name, child in list(module._modules.items()):
                if child is not None: setattr(module, name, rewrite(child))
        return result
    return rewrite(model)


parameter_groups = core.parameter_groups


def assert_no_clamp(model):
    return {name: module.assert_no_clamp() for name, module in model.named_modules() if isinstance(module, AblationLinear)}


def backend(spec, factor_seed=42, *, factor_initializer=None, data_spec=None):
    spec = _spec(spec)
    from experiments.adapters import ADAPTER_TYPES
    return SimpleNamespace(adapter_types=(core.AblationLinear, *ADAPTER_TYPES),
                           inject_adapters=lambda model, method, rank=8, targets=None: inject_adapters(model, spec, rank, targets, factor_seed, method,factor_initializer,data_spec),
                           recipe_manifest=lambda method: recipe_manifest(spec, method,data_spec),
                           parameter_groups=parameter_groups, assert_no_clamp=assert_no_clamp)
