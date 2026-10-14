#!/usr/bin/env python3
"""Protocol-v2 native run with sealed IDs, exact token manifests, and group loss."""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rankstudy.adapters import inject, modules
from rankstudy.allocator import Slot, allocate
from rankstudy.registry import build_registry
from rankstudy.scores import (activation_scores, adalora_scores, displacement_scores,
                              ema_update, raw_scores, uncertainty_update)
from rankstudy.v2.data import collate, microbatches, read_jsonl


def loss_parts(logits, labels):
    pred = logits[..., :-1, :].float()
    target = labels[..., 1:]
    per = torch.nn.functional.cross_entropy(pred.transpose(-1, -2), target,
                                            ignore_index=-100, reduction="none")
    valid = target.ne(-100)
    return (per * valid).sum(), valid.sum()


@torch.no_grad()
def evaluate_group_loss(model, rows, tokenizer, device):
    model.eval()
    values = []
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(rows), 2):
        batch = collate(rows[start:start + 2], tokenizer.pad_token_id, device)
        # Dev rows are one deterministic representative per group. Keep the
        # per-example numerator/denominator so group balancing is explicit.
        pred = model(**batch).logits[..., :-1, :].float()
        target = batch["labels"][..., 1:]
        tok_loss = torch.nn.functional.cross_entropy(pred.transpose(-1, -2), target,
                                                     ignore_index=-100, reduction="none")
        valid = target.ne(-100)
        for i in range(pred.shape[0]):
            numerator = (tok_loss[i] * valid[i]).sum()
            count = valid[i].sum().clamp_min(1)
            values.append(float(numerator / count))
            total_loss += float(numerator)
            total_tokens += int(count)
    model.train()
    group_balanced = sum(values) / max(1, len(values))
    return {"group_balanced_nll": group_balanced,
            "token_weighted_nll": total_loss / max(1, total_tokens)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--init", choices=["L", "I", "T0", "T1", "ADALORA"], default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--updates", type=int, default=8)
    ap.add_argument("--data-root", default="/var/tmp/dora-adaptive-rank/data_v2")
    ap.add_argument("--manifest", default="artifacts/manifests_v2")
    ap.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--target-rank", type=int, default=8)
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--microbatch", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    manifest_path = Path(args.manifest) / f"train_seed{args.seed}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if args.updates > len(manifest["updates"]):
        raise ValueError(f"manifest has {len(manifest['updates'])} updates, requested {args.updates}")
    # Short HPO trials consume the prefix of the same sealed paired stream;
    # full-horizon runs consume all updates. Keeping one manifest preserves
    # identical row ordering and token accounting across both stages.
    manifest["updates"] = manifest["updates"][:args.updates]
    train = read_jsonl(Path(args.data_root) / "train.jsonl")
    dev = read_jsonl(Path(args.data_root) / "dev_tune.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    registry = {x.id: x for x in build_registry()}
    spec = registry.get(args.method)
    init = args.init or (spec.init if spec else "L")
    adalora = args.method == "ADALORA_REF" or init == "ADALORA"
    triplet = init in ("T0", "T1")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").to(device)
    model = inject(model, args.rank, ("q_proj", "o_proj"), init, args.scale, args.method,
                   args.seed, triplet, penalty=args.penalty, adalora=adalora)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    warmup = max(1, int(round(0.03 * args.updates)))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        q = (step - warmup) / max(1, args.updates - warmup - 1)
        return 0.10 + 0.90 * 0.5 * (1 + torch.cos(torch.tensor(q * torch.pi))).item()

    for group in opt.param_groups:
        group["lr"] = args.lr * lr_factor(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_bytes = (json.dumps(vars(args), sort_keys=True, indent=2) + "\n").encode()
    (out / "config.json").write_bytes(config_bytes)

    metrics, score_ema, uncertainty = [], {}, {}
    t0 = time.time()
    model.train()
    for step, update in enumerate(manifest["updates"]):
        for module in modules(model):
            module.begin_score_accum()
        opt.zero_grad(set_to_none=True)
        input_tokens = supervised_tokens = 0
        task_loss_sum = 0.0
        for indices in microbatches(update["row_indices"], train, args.microbatch):
            batch_rows = [train[i] for i in indices]
            batch = collate(batch_rows, tokenizer.pad_token_id, device)
            valid_x = batch["attention_mask"].detach().bool()
            for module in modules(model):
                module._score_valid.append(valid_x.reshape(-1))
            output = model(**batch)
            part, count = loss_parts(output.logits, batch["labels"])
            part.backward()
            input_tokens += int(batch["attention_mask"].sum())
            supervised_tokens += int(count)
            task_loss_sum += float(part.detach())
        denom = max(1, supervised_tokens)
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.div_(denom)
        for module in modules(model):
            if module._gate_grad_accum is not None:
                module._gate_grad_accum.div_(denom)

        pending_keep = None
        progress = (step + 1) / args.updates
        # Custom allocators use the declared pre-optimizer global-batch
        # gradients.  Faithful AdaLoRA is handled after optimizer.step below,
        # matching the upstream trainer ordering and allowing masked E
        # coefficients to reactivate between allocation events.
        if (spec and spec.allocator != "none" and not any(m.adalora for m in modules(model))
                and ((step + 1) % 16 == 0 or step == args.updates - 1)):
            allslots = []
            for mi, module in enumerate(modules(model)):
                if spec.allocator == "global_random":
                    q = torch.tensor([random.Random(args.seed * 1000003 + mi * 1009 + j).random()
                                      for j in range(module.rank)], device=device)
                elif spec.score == "ADALORA":
                    q = adalora_scores(module.adapter)
                elif spec.score == "RAW":
                    q = raw_scores(module.adapter)
                elif spec.score == "DISP":
                    q = displacement_scores(module.adapter)
                elif spec.score == "ACT":
                    xs = torch.cat(module._score_x, dim=0)
                    valid = torch.cat([v for v in module._score_valid if v is not None], dim=0)
                    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
                    if valid_idx.numel() > 256:
                        generator = torch.Generator(device="cpu").manual_seed(args.seed * 1009 + (step + 1) * 9176 + mi)
                        valid_idx = valid_idx[torch.randperm(valid_idx.numel(), generator=generator)[:256].to(valid_idx.device)]
                        xs, valid = xs[valid_idx], None
                    q = activation_scores(module.adapter, xs, valid)
                else:
                    q = module.score_gate()
                raw_q = q
                smoothed_q = ema_update(score_ema.get(mi), raw_q, 0.85)
                score_ema[mi] = smoothed_q
                if "UNCERTAINTY" in args.method or module.adalora:
                    uncertainty[mi] = uncertainty_update(uncertainty.get(mi), raw_q, smoothed_q, 0.85)
                    q = smoothed_q * uncertainty[mi]
                else:
                    q = smoothed_q
                cost = module.A.shape[1] + module.B.shape[0] + int(module.triplet or module.adalora)
                for j, value in enumerate(q.tolist()):
                    if not module.adalora and not bool(module.active_mask[j]):
                        continue
                    allslots.append(Slot(str(mi), j, cost, float(value)))
            initial = sum(m.rank * (m.A.shape[1] + m.B.shape[0] + int(m.triplet or m.adalora)) for m in modules(model))
            target = sum(args.target_rank * (m.A.shape[1] + m.B.shape[0] + int(m.triplet or m.adalora)) for m in modules(model))
            u = max(0.0, min(1.0, (progress - 0.10) / 0.70))
            budget = round(target + (initial - target) * (1 - u) ** 3)
            floor = {} if any(m.adalora for m in modules(model)) else {str(i): 1 for i in range(len(modules(model)))}
            if spec.allocator == "uniform_gate" and len({s.cost for s in allslots}) == 1:
                cost = allslots[0].cost
                slots = min(sum(m.rank for m in modules(model)), max(0, budget // cost))
                base, extra = divmod(slots, len(modules(model)))
                keep = set()
                for mi, module in enumerate(modules(model)):
                    group = [s for s in allslots if s.module == str(mi)]
                    keep.update((s.module, s.index) for s in sorted(group, key=lambda s: (-s.score, s.index))[:base + int(mi < extra)])
            else:
                keep = allocate(allslots, budget, floor)
            for mi, module in enumerate(modules(model)):
                module.set_mask(torch.tensor([(str(mi), j) in keep for j in range(module.rank)], device=device))

        regularizer = torch.zeros((), device=device)
        if spec and spec.triplet and args.penalty:
            regularizer = args.penalty * sum(m.orth_penalty() for m in modules(model)) / max(1, len(modules(model)))
        elif any(m.adalora for m in modules(model)):
            regularizer = 0.1 * sum(m.adalora_orth_penalty() for m in modules(model)) / max(1, len(modules(model)))
        if regularizer.requires_grad:
            regularizer.backward()
        for module in modules(model):
            mask = module.active_mask.float()
            if not module.adalora:
                if module.A.grad is not None: module.A.grad.mul_(mask[:, None])
                if module.B.grad is not None: module.B.grad.mul_(mask[None, :])
            if hasattr(module, "lam") and module.lam.grad is not None:
                module.lam.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # Faithful AdaLoRA reference path.  Upstream updates element scores
        # after optimizer.step using the post-step parameter and retained
        # gradient, smooths every step through the shrink interval, and only
        # masks at allocation events.  Crucially, E is zeroed when a slot is
        # masked but is then free to move with Adam between events, preserving
        # the reference reactivation semantics.  The q/o screen has equal
        # module costs, so this is the exact global slot budget in rank units.
        if any(m.adalora for m in modules(model)):
            for mi, module in enumerate(modules(model)):
                if progress < 0.80:
                    raw_q = adalora_scores(module.adapter)
                    smoothed_q = ema_update(score_ema.get(mi), raw_q, 0.85)
                    score_ema[mi] = smoothed_q
                    uncertainty[mi] = uncertainty_update(uncertainty.get(mi), raw_q, smoothed_q, 0.85)
            if ((step + 1) % 16 == 0 or step == args.updates - 1) and progress > 0.10:
                allslots = []
                for mi, module in enumerate(modules(model)):
                    q = score_ema.get(mi, torch.zeros(module.rank, device=device)) * uncertainty.get(
                        mi, torch.zeros(module.rank, device=device))
                    for j, value in enumerate(q.tolist()):
                        allslots.append(Slot(str(mi), j, 1, float(value)))
                initial_slots = sum(m.rank for m in modules(model))
                target_slots = args.target_rank * len(modules(model))
                u = max(0.0, min(1.0, (progress - 0.10) / 0.70))
                budget_slots = round(target_slots + (initial_slots - target_slots) * (1 - u) ** 3)
                keep = allocate(allslots, budget_slots, {})
                for mi, module in enumerate(modules(model)):
                    module.set_mask(torch.tensor([(str(mi), j) in keep for j in range(module.rank)], device=device))
        if pending_keep is not None:
            for mi, module in enumerate(modules(model)):
                module.set_mask(torch.tensor([(str(mi), j) in pending_keep for j in range(module.rank)], device=device))
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_factor(step + 1)
        for module in modules(model):
            mask = module.active_mask.float()
            states = [] if module.adalora else [(module.A, mask[:, None]), (module.B, mask[None, :])] + ([(module.lam, mask)] if hasattr(module, "lam") else [])
            for parameter, view in states:
                state = opt.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state: state[key].mul_(view)
        metrics.append({"step": step + 1, "task_loss": task_loss_sum / denom,
                        "regularizer": float(regularizer.detach()), "lr": opt.param_groups[0]["lr"],
                        "input_tokens": input_tokens, "supervised_tokens": supervised_tokens,
                        "active_slots": sum(int(m.active_mask.sum()) for m in modules(model)),
                        "elapsed_s": time.time() - t0})
        print(json.dumps(metrics[-1]), flush=True)

    dev = evaluate_group_loss(model, dev, tokenizer, device)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {"protocol": "adaptive-rank-v2", "method": args.method, "init": init,
                "lr": args.lr, "scale": args.scale, "penalty": args.penalty,
                "rank": args.rank, "target_rank": args.target_rank,
                "seed": args.seed, "updates": args.updates, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_revision": source_revision, "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(), "status": "complete",
                "final_active_slots": metrics[-1]["active_slots"],
                "input_tokens": sum(x["input_tokens"] for x in metrics),
                "supervised_tokens": sum(x["supervised_tokens"] for x in metrics),
                "elapsed_s": metrics[-1]["elapsed_s"], "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "dev_tune_endpoint_group_balanced_nll": dev["group_balanced_nll"],
                "dev_tune_endpoint_token_weighted_nll": dev["token_weighted_nll"],
                "adalora_update_order": "post_optimizer_reference" if any(m.adalora for m in modules(model)) else None}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
