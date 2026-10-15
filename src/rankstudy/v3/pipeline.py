"""Pure stage planning; pipeline work is charged even when cached physically."""
import hashlib
import random


def export_policy(method, common_deployment=False):
    """The common pipeline applies the same operator to all source families."""
    return 'global' if common_deployment or 'GLOBAL_SVD' in method else 'uniform'


def stage_seed(seed,stage):
    return int.from_bytes(hashlib.sha256(f'{seed}/{stage}'.encode()).digest()[:8],'little')%(2**63)


def token_boundary(updates,fraction):
    """First intact update crossing the target, with exact overshoot reported."""
    if not 0<fraction<1: raise ValueError('fraction must lie inside horizon')
    total=sum(u['input_tokens'] for u in updates); consumed=0
    for i,u in enumerate(updates,1):
        consumed+=u['input_tokens']
        if consumed>=fraction*total:
            if i>=len(updates): raise ValueError('no recovery update remains')
            return i,{'target_fraction':fraction,'actual_fraction':consumed/total,
                      'target_tokens':fraction*total,'actual_tokens':consumed,
                      'overshoot_tokens':consumed-fraction*total}


def shuffle_compatible(pattern,shapes,seed):
    """Shuffle ranks within exact input/output shape classes, never across costs."""
    groups={}
    for name in sorted(pattern): groups.setdefault(tuple(shapes[name]),[]).append(name)
    rng=random.Random(seed); result={}
    for shape,names in sorted(groups.items()):
        values=[pattern[n] for n in names]; rng.shuffle(values)
        result.update(zip(names,values))
    return result
