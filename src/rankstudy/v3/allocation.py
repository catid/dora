"""Token-progress schedules and deterministic scalar-cost allocation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class Candidate:
    module: str
    slot: int
    cost: int
    value: float


def allocate(candidates, budget, floor=1):
    """Exact equal-cost top-k or small DP; otherwise shared value/cost heuristic.

    Costs must be constant within a module. Return the selection AND algorithm,
    spend and underspend; a heuristic is never labeled optimal.
    """
    items = sorted(candidates, key=lambda x: (x.module, x.slot))
    if any(c.cost <= 0 or not math.isfinite(c.value) for c in items):
        raise ValueError("nonpositive cost or nonfinite score")
    groups = {name: [c for c in items if c.module == name] for name in sorted({c.module for c in items})}
    chosen = []
    for name, group in groups.items():
        if len({c.cost for c in group}) != 1:
            raise ValueError("slot costs must be constant within each module")
        if len(group) < floor:
            raise ValueError("floor exceeds available slots")
        chosen.extend(sorted(group, key=lambda c: (-c.value, c.slot))[:floor])
    required = sum(c.cost for c in chosen)
    if required > budget:
        raise ValueError("infeasible floor")
    selected = {(c.module, c.slot) for c in chosen}
    remaining = [c for c in items if (c.module, c.slot) not in selected]
    room = budget - required
    gcd = reduce(math.gcd, (c.cost for c in items), 1 if not items else items[0].cost)
    if len({c.cost for c in items}) <= 1:
        algorithm = "equal_cost_exact_topk"
        take = min(len(remaining), room // items[0].cost) if items else 0
        chosen += sorted(remaining, key=lambda c: (-c.value, c.module, c.slot))[:take]
    elif room // gcd <= 8192:
        algorithm = "exact_integer_dp"
        dp = {0: (0., ())}
        for j, c in enumerate(remaining):
            for used, (value, ids) in list(dp.items()):
                nxt = used + c.cost // gcd
                if nxt > room // gcd:
                    continue
                proposal = (value + c.value, ids + (j,))
                old = dp.get(nxt)
                if old is None or proposal[0] > old[0] or (proposal[0] == old[0] and proposal[1] < old[1]):
                    dp[nxt] = proposal
        # Max value, then max utilization, then lexicographically smallest IDs.
        used, (_, ids) = min(dp.items(), key=lambda kv: (-kv[1][0], -kv[0], kv[1][1]))
        chosen += [remaining[j] for j in ids]
    else:
        algorithm = "value_per_cost_greedy"
        for c in sorted(remaining, key=lambda c: (-c.value/c.cost, c.module, c.slot)):
            if c.cost <= room:
                chosen.append(c)
                room -= c.cost
    spend = sum(c.cost for c in chosen)
    return {(c.module, c.slot) for c in chosen}, {"algorithm": algorithm, "budget": budget,
                                                   "spent": spend, "underspend": budget-spend}


def uniform_allocate(candidates, budget, floor=1):
    """Keep ranks as uniform as feasible, then within-module score order."""
    groups = {}
    for c in candidates:
        groups.setdefault(c.module, []).append(c)
    for name in groups:
        groups[name].sort(key=lambda c: (-c.value, c.slot))
    ranks = {name: floor for name in groups}
    if any(len(group) < floor for group in groups.values()):
        raise ValueError("infeasible floor")
    selected = [c for group in groups.values() for c in group[:floor]]
    used = sum(c.cost for c in selected)
    if used > budget:
        raise ValueError("infeasible floor")
    while True:
        choices = [name for name, group in groups.items()
                   if ranks[name] < len(group) and used + group[ranks[name]].cost <= budget]
        if not choices:
            break
        name = min(choices, key=lambda name: (ranks[name], name))
        c = groups[name][ranks[name]]
        selected.append(c)
        ranks[name] += 1
        used += c.cost
    return {(c.module, c.slot) for c in selected}, {
        "algorithm": "uniform_ranks_within_module_score", "budget": budget,
        "spent": used, "underspend": budget-used}


class TokenSchedule:
    def __init__(self, tokens_per_update, initial, target, warmup=.10, freeze=.80, interval=16):
        if not tokens_per_update or any(t <= 0 for t in tokens_per_update):
            raise ValueError("positive token manifest required")
        if not 0 <= warmup < freeze < 1:
            raise ValueError("invalid fractions")
        self.total = sum(tokens_per_update)
        self.initial, self.target = initial, target
        self.warmup, self.freeze, self.interval = warmup, freeze, interval
        cumulative = 0
        self.freeze_update = None
        for i, count in enumerate(tokens_per_update, 1):
            cumulative += count
            if cumulative >= freeze*self.total:
                self.freeze_update = i
                break
        if self.freeze_update >= len(tokens_per_update):
            raise ValueError("manifest leaves no post-allocation recovery update")

    def at(self, update, cumulative_tokens):
        progress = cumulative_tokens/self.total
        u = min(1., max(0., (progress-self.warmup)/(self.freeze-self.warmup)))
        budget = math.floor(self.target + (self.initial-self.target)*(1-u)**3)
        event = (progress > self.warmup and update <= self.freeze_update
                 and (update % self.interval == 0 or update == self.freeze_update))
        return budget, event, update == self.freeze_update
