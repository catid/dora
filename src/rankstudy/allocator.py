"""Deterministic budget schedules and allocation helpers."""
from __future__ import annotations
from dataclasses import dataclass
import itertools


def cubic_budget(step, total_steps, initial_cost, target_cost, warmup=.10, freeze=.80):
    if not (0 <= warmup < freeze <= 1): raise ValueError('invalid schedule')
    p = step/max(1,total_steps)
    if p <= warmup: return int(initial_cost), False
    if p >= freeze: return int(target_cost), True
    u=(p-warmup)/(freeze-warmup)
    return int(round(target_cost+(initial_cost-target_cost)*(1-u)**3)), step % 16 == 0

@dataclass(frozen=True)
class Slot:
    module: str
    index: int
    cost: int
    score: float


def exact_knapsack(slots, budget, minimum_by_module=None):
    """Max-score <= budget for small integer costs, stable ID tie-breaking."""
    slots=list(slots); minimum_by_module=minimum_by_module or {}
    mods=sorted({s.module for s in slots})
    required={m:minimum_by_module.get(m,0) for m in mods}
    base=[]; remaining=[]
    for m in mods:
        group=sorted((s for s in slots if s.module==m), key=lambda s:(-s.score,s.index))
        if len(group)<required[m]: raise ValueError('minimum exceeds available slots')
        base += group[:required[m]]; remaining += group[required[m]:]
    base_cost=sum(s.cost for s in base)
    if base_cost>budget: raise ValueError('budget below floor')
    # DP stores score and tuple IDs; maximize score, then lexicographically smallest IDs.
    dp={0:(0.0,())}
    for s in sorted(remaining,key=lambda x:(x.module,x.index)):
        for c,(v,ids) in list(dp.items())[::-1]:
            nc=c+s.cost
            if nc>budget-base_cost: continue
            cand=(v+s.score,ids+((s.module,s.index),))
            old=dp.get(nc)
            if old is None or cand[0]>old[0] or (cand[0]==old[0] and cand[1]<old[1]): dp[nc]=cand
    best=max(dp.items(),key=lambda kv:(kv[1][0],-kv[0],tuple(reversed(kv[1][1]))))
    selected={(s.module,s.index) for s in base}
    selected.update(best[1][1])
    return selected, base_cost + best[0], best[1][0]


def allocate(slots,budget,minimum_by_module=None):
    slots=list(slots); minimum_by_module=minimum_by_module or {}
    if not slots:return set()
    costs={s.cost for s in slots}
    # Exact DP is used for feasible small cases. Large model costs use a stable
    # value-per-cost fallback; this is explicitly an approximation.
    if max(costs,default=0)<=64 and budget<=4096:
        selected,_,score=exact_knapsack(slots,budget,minimum_by_module)
        return selected
    groups={}
    for s in slots: groups.setdefault(s.module,[]).append(s)
    selected=set(); used=0
    for m,group in sorted(groups.items()):
        req=minimum_by_module.get(m,0)
        for s in sorted(group,key=lambda x:(-x.score/x.cost if x.cost else float('inf'),x.index))[:req]: selected.add((s.module,s.index)); used+=s.cost
    if used>budget: raise ValueError('budget below floor')
    for s in sorted(slots,key=lambda x:(-(x.score/x.cost if x.cost else float('inf')),x.module,x.index)):
        if (s.module,s.index) in selected or used+s.cost>budget: continue
        selected.add((s.module,s.index)); used+=s.cost
    return selected
