"""Explicit normalization of task numerators and non-parameter score state.

Call after all effective-batch backwards, before score absolute values, clipping,
regularization or optimizer.step. DDP has already averaged parameter gradients;
unregistered gate gradients and ACT sums have not been reduced by DDP.
"""
import torch
import torch.distributed as dist
from .adapters import adapted


@torch.no_grad()
def normalize_task_gradients(model, local_supervised_tokens, ddp_averaged=False):
    module=model.module if isinstance(model,torch.nn.parallel.DistributedDataParallel) else model
    params=[p for p in module.parameters() if p.requires_grad]
    denominator=torch.tensor(float(local_supervised_tokens),device=params[0].device,dtype=torch.float64)
    distributed=dist.is_available() and dist.is_initialized()
    world=dist.get_world_size() if distributed else 1
    if distributed:
        dist.all_reduce(denominator,op=dist.ReduceOp.SUM)
    if denominator.item()<=0:
        raise ValueError('global effective batch has no supervised tokens')
    for p in params:
        if p.grad is None:
            # Explicit zeros retain collective order if a module is unused.
            p.grad=torch.zeros_like(p)
        if distributed and not ddp_averaged:
            dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
        p.grad.mul_((world if ddp_averaged else 1)/denominator)
    for _,m in sorted(adapted(module).items()):
        if m.tracking and m.score=='GATE':
            if m._gate.grad is None:
                m._gate.grad=torch.zeros_like(m._gate)
            if distributed:
                dist.all_reduce(m._gate.grad,op=dist.ReduceOp.SUM)
        if m.tracking and m.score=='ACT':
            if m._act_sum is None:
                m._act_sum=torch.zeros(m.rank,device=denominator.device)
            count=torch.tensor(m._act_count,device=denominator.device,dtype=torch.int64)
            if distributed:
                dist.all_reduce(m._act_sum,op=dist.ReduceOp.SUM)
                dist.all_reduce(count,op=dist.ReduceOp.SUM)
            m._act_count=int(count)
    return int(denominator)


@torch.no_grad()
def broadcast_masks(model, optimizer, masks_on_rank_zero):
    module=model.module if isinstance(model,torch.nn.parallel.DistributedDataParallel) else model
    distributed=dist.is_available() and dist.is_initialized()
    rank=dist.get_rank() if distributed else 0
    for name,m in sorted(adapted(module).items()):
        mask=(masks_on_rank_zero[name].to(device=m.mask.device,dtype=torch.uint8) if rank==0
              else torch.zeros_like(m.mask,dtype=torch.uint8))
        if distributed: dist.broadcast(mask,src=0)
        m.set_mask(mask.bool(),optimizer)
