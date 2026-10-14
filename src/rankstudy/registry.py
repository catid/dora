"""Declarative 41-cell method registry."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import itertools,yaml

@dataclass(frozen=True)
class Method:
 id:str; group:str; init:str; allocator:str='none'; score:str='none'; triplet:bool=False; penalty:float=0.0; pipeline:str='native'; adalora:bool=False

def build_registry():
 out=[Method('L_FIXED','fixed','L'),Method('I_FIXED','fixed','I'),Method('L_WIDE','wide','L'),Method('I_WIDE','wide','I'),Method('ADALORA_REF','reference','ADALORA','adalora','ADALORA',False,0.0,'native',True)]
 for init in ('L','I'):
  for score in ('RAW','GATE','DISP','ACT'): out.append(Method(f'{init}_{score}','online',init,'monotonic',score))
  out += [Method(f'{init}_UNIFORM_GATE','allocation_control',init,'uniform_gate','GATE'),Method(f'{init}_GLOBAL_RANDOM','allocation_control',init,'global_random')]
 for init in ('T0','T1'):
  out += [Method(f'{init}_FIXED','triplet',init,triplet=True),Method(f'{init}_GATE','triplet',init,'monotonic','GATE',True)]
  for suffix in ('FIXED_ORTHO','GATE_ORTHO'):
   out.append(Method(f'{init}_{suffix}','triplet_ortho',init,'monotonic' if 'GATE' in suffix else 'none','GATE' if 'GATE' in suffix else 'none',True,0.001))
 for pi,st in itertools.product(('L','I'),repeat=2): out.append(Method(f'{pi}_PILOT_{st}_STUDENT','restart',st,'pilot_gate', 'GATE'))
 for st in ('L','I'): out += [Method(f'{st}_UNIFORM_RESTART','restart_control',st,'uniform_restart'),Method(f'{st}_SHUFFLED_SELF_RESTART','restart_control',st,'shuffled_self')]
 for init in ('L','I'):
  for pol in ('UNIFORM_SVD','GLOBAL_SVD'): out.append(Method(f'{init}_{pol}','compress',init,pipeline='compress'))
 for init in ('L','I'):
  for score in ('RAW','GATE'): out.append(Method(f'{init}_{score}_UNCERTAINTY','online_uncertainty',init,'monotonic',score))
 # The registry above intentionally includes all required leaf controls.
 assert len(out)==41, len(out)
 return out

def write_yaml(path):
 path.write_text(yaml.safe_dump({'schema':'adaptive-rank-method-registry-v1','methods':[asdict(x) for x in build_registry()]},sort_keys=False))
