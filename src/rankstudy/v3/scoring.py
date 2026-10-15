"""One pre-optimizer score update per effective global batch."""
from __future__ import annotations

import torch


class SlotEMA:
    """Zero-start EMA, no bias correction; q is the SAME batch-mean task score."""
    def __init__(self, beta=.85, beta_u=.85, uncertainty=False):
        self.beta, self.beta_u, self.uncertainty = beta, beta_u, uncertainty
        self.mean, self.deviation = {}, {}
        self.updates = 0

    @torch.no_grad()
    def update(self, scores):
        for name, q in scores.items():
            q = q.detach().float()
            old = self.mean.get(name, torch.zeros_like(q))
            self.mean[name] = self.beta*old + (1-self.beta)*q
            if self.uncertainty:
                old_u = self.deviation.get(name, torch.zeros_like(q))
                self.deviation[name] = (self.beta_u*old_u
                    + (1-self.beta_u)*(q-self.mean[name]).abs())
        self.updates += 1

    def value(self, name):
        return self.mean[name] * self.deviation[name] if self.uncertainty else self.mean[name]

    def state_dict(self):
        return dict(beta=self.beta, beta_u=self.beta_u, uncertainty=self.uncertainty,
                    mean=self.mean, deviation=self.deviation, updates=self.updates)

    def load_state_dict(self, state):
        self.__dict__.update(state)


def token_subsample(attention_masks, seed, update, max_tokens=256):
    """Fixed valid-token sample independent of method, rank and microbatch count.

    Masks are given in sealed example order. Padding never enters the universe.
    The returned local flat indices can be reused at every adapted module.
    """
    counts = [int(mask.sum()) for mask in attention_masks]
    generator = torch.Generator().manual_seed(seed*1000003 + update)
    selected = torch.randperm(sum(counts), generator=generator)[:max_tokens].sort().values
    out, offset = [], 0
    for mask, count in zip(attention_masks, counts):
        valid = torch.nonzero(mask.reshape(-1).cpu()).flatten()
        local = selected[(selected >= offset) & (selected < offset+count)] - offset
        out.append(valid[local])
        offset += count
    return out
