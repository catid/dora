#!/usr/bin/env python3
"""Run the registered restart or train/compress/recover pipeline.

This is intentionally separate from the native runner: both multi-stage
pipelines must charge discarded pilot or compression work and must reload the
unchanged original backbone before the student/recovery phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rankstudy.adapters import StudyLinear, inject, modules  # noqa: E402
from rankstudy.allocator import Slot, allocate  # noqa: E402
from rankstudy.compress import compress_centered, thin_core_svd  # noqa: E402
from rankstudy.v2.data import collate, microbatches, read_jsonl  # noqa: E402


def loss_parts(logits, labels):
    pred = logits[..., :-1, :].float()
    target = labels[..., 1:]
    per = torch.nn.functional.cross_entropy(pred.transpose(-1, -2), target,
                                            ignore_index=-100, reduction="none")
    valid = target.ne(-100)
    return (per * valid).sum(), valid.sum()


@torch.no_grad()
def evaluate(model, rows, tokenizer, device):
    model.eval()
    group_values, total, count = [], 0.0, 0
    for start in range(0, len(rows), 2):
        batch = collate(rows[start:start + 2], tokenizer.pad_token_id, device)
        logits = model(**batch).logits.float()
        target = batch["labels"][..., 1:]
        per = torch.nn.functional.cross_entropy(logits[..., :-1, :].transpose(-1, -2), target,
                                                ignore_index=-100, reduction="none")
        valid = target.ne(-100)
        for i in range(target.shape[0]):
            n = valid[i].sum().clamp_min(1)
            group_values.append(float((per[i] * valid[i]).sum() / n))
            total += float((per[i] * valid[i]).sum())
            count += int(n)
    model.train()
    return {"group_balanced_nll": sum(group_values) / max(1, len(group_values)),
            "token_weighted_nll": total / max(1, count)}


def inject_pattern(model, pattern, init, scale, method, seed, target_modules=("q_proj", "o_proj")):
    """Inject a fresh adapter with an independently selected rank per module."""
    model.requires_grad_(False)
    selected = []
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and any(name == t or name.endswith("." + t) for t in target_modules):
            selected.append((name, module))
    if not selected:
        raise ValueError("no target modules")
    for name, base in selected:
        parent = model
        bits = name.split(".")
        for bit in bits[:-1]:
            parent = getattr(parent, bit)
        stable = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        rank = int(pattern[name])
        setattr(parent, bits[-1], StudyLinear(base, rank, init, scale, method,
                                               seed + stable % 100000).to(base.weight.device))
    return model


def load_model(path, dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype,
                                                attn_implementation="sdpa").to("cuda")


def optimizer(model, lr):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)


def set_lr(opt, peak, step, total):
    warmup = max(1, int(round(.03 * total)))
    if step < warmup:
        frac = (step + 1) / warmup
    else:
        q = (step - warmup) / max(1, total - warmup - 1)
        frac = .10 + .90 * .5 * (1 + math.cos(math.pi * q))
    for group in opt.param_groups:
        group["lr"] = peak * frac


def mask_by_gate(model, target_cost, initial_cost, progress, seed, floor=True):
    allslots = []
    for mi, module in enumerate(modules(model)):
        q = module.score_gate()
        cost = module.A.shape[1] + module.B.shape[0]
        for j, value in enumerate(q.tolist()):
            if bool(module.active_mask[j]):
                allslots.append(Slot(str(mi), j, cost, float(value)))
    u = max(0.0, min(1.0, (progress - .10) / .70))
    budget = round(target_cost + (initial_cost - target_cost) * (1 - u) ** 3)
    minimum = {str(i): 1 for i in range(len(modules(model)))} if floor else {}
    keep = allocate(allslots, budget, minimum)
    for mi, module in enumerate(modules(model)):
        module.set_mask(torch.tensor([(str(mi), j) in keep for j in range(module.rank)],
                                      device=module.active_mask.device))


def train_updates(model, updates, train_rows, tokenizer, peak_lr, microbatch, total_steps,
                  start_step=0, allocate_gate=False, allocation_total_steps=None):
    device = "cuda"
    opt = optimizer(model, peak_lr)
    metrics, t0 = [], time.time()
    if allocation_total_steps is None:
        allocation_total_steps = total_steps
    initial_cost = sum(m.rank * (m.A.shape[1] + m.B.shape[0]) for m in modules(model))
    target_cost = sum(max(1, m.rank // 2) * (m.A.shape[1] + m.B.shape[0]) for m in modules(model))
    model.train()
    for local_step, update in enumerate(updates):
        step = start_step + local_step
        for module in modules(model):
            module.begin_score_accum()
        opt.zero_grad(set_to_none=True)
        input_tokens = supervised_tokens = 0
        task_loss_sum = 0.0
        for indices in microbatches(update["row_indices"], train_rows, microbatch):
            rows = [train_rows[i] for i in indices]
            batch = collate(rows, tokenizer.pad_token_id, device)
            output = model(**batch)
            part, count = loss_parts(output.logits, batch["labels"])
            part.backward()
            input_tokens += int(batch["attention_mask"].sum())
            supervised_tokens += int(count)
            task_loss_sum += float(part.detach())
        denom = max(1, supervised_tokens)
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(denom)
        progress = (local_step + 1) / max(1, allocation_total_steps)
        if allocate_gate and ((local_step + 1) % 16 == 0 or local_step == len(updates) - 1):
            mask_by_gate(model, target_cost, initial_cost, progress, 0)
        # Apply the newly persistent mask before the update. This prevents the
        # allocation event itself from giving a pruned coordinate one last
        # Adam/momentum step.
        for module in modules(model):
            mask = module.active_mask.float()
            if module.A.grad is not None: module.A.grad.mul_(mask[:, None])
            if module.B.grad is not None: module.B.grad.mul_(mask[None, :])
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        set_lr(opt, peak_lr, local_step, total_steps)
        opt.step()
        # Persistent pruning: remove both gradient and optimizer-state drift.
        for module in modules(model):
            mask = module.active_mask.float()
            for p, view in ((module.A, mask[:, None]), (module.B, mask[None, :])):
                state = opt.state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key].mul_(view)
        metrics.append({"step": step + 1, "task_loss": task_loss_sum / denom,
                        "input_tokens": input_tokens, "supervised_tokens": supervised_tokens,
                        "active_slots": sum(int(m.active_mask.sum()) for m in modules(model)),
                        "elapsed_s": time.time() - t0})
    return metrics


def pattern_from_masks(model):
    return {name: int(module.active_mask.sum()) for name, module in
            ((name, module) for name, module in model.named_modules() if isinstance(module, StudyLinear))}


def module_pattern(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, StudyLinear)}


def global_svd_factors(model, target_rank):
    entries, total = [], 0
    for name, module in module_pattern(model).items():
        left, s, right, meta = thin_core_svd(module.A0, module.B0, module.A, module.B, module.scale)
        entries.append((name, module, left, s, right, meta))
        total += target_rank
    # Equal-cost q/o modules: global energy selection is the exact prefix-feasible
    # solution for the specified Frobenius objective at this budget.
    chosen = {name: target_rank for name, *_ in entries}
    remaining = total
    ranked = sorted(((float(sv * sv), name, j) for name, _, _, s, _, _ in entries for j, sv in enumerate(s)), reverse=True)
    chosen = {name: 0 for name, *_ in entries}
    for _, name, j in ranked:
        if remaining <= 0:
            break
        if j == chosen[name]:
            chosen[name] += 1
            remaining -= 1
    return entries, chosen


def export_factors(entries, ranks):
    out = {}
    for name, module, left, s, right, meta in entries:
        r = int(ranks[name])
        root = s[:r].clamp_min(0).sqrt()
        out[name] = {"B": (left[:, :r] * root[None, :]).float().cpu(),
                     "A": (right[:, :r] * root[None, :]).T.float().cpu(),
                     "rank": r, "meta": meta}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--pilot-init", default=None)
    ap.add_argument("--student-init", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--updates", type=int, default=512)
    ap.add_argument("--data-root", default="/var/tmp/dora-adaptive-rank/data_v2")
    ap.add_argument("--manifest", default="artifacts/manifests_v2")
    ap.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--target-rank", type=int, default=8)
    ap.add_argument("--microbatch", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    manifest_path = Path(args.manifest) / f"train_seed{args.seed}.json"
    manifest = json.loads(manifest_path.read_text())
    if args.updates > len(manifest["updates"]):
        raise ValueError("requested updates exceed manifest")
    updates = manifest["updates"][:args.updates]
    train = read_jsonl(Path(args.data_root) / "train.jsonl")
    dev = read_jsonl(Path(args.data_root) / "dev_tune.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"; tokenizer.pad_token = tokenizer.eos_token
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); source_model = None; all_metrics = []
    is_restart = "PILOT" in args.method or "RESTART" in args.method
    is_compress = "SVD" in args.method
    if is_restart:
        pilot_init = args.pilot_init or args.method.split("_PILOT_")[0]
        student_init = args.student_init or args.init or args.method.split("_STUDENT")[-1]
        if student_init not in ("L", "I"):
            student_init = "I" if "_I_STUDENT" in args.method else "L"
        source_model = load_model(args.model)
        source_model = inject(source_model, args.rank, ("q_proj", "o_proj"), pilot_init, args.scale, args.method, args.seed)
        pilot_steps = max(1, int(round(.20 * args.updates)))
        all_metrics += train_updates(source_model, updates[:pilot_steps], train, tokenizer, args.lr,
                                     args.microbatch, pilot_steps, allocate_gate=True,
                                     allocation_total_steps=pilot_steps)
        pattern = pattern_from_masks(source_model)
        pilot_tokens = sum(x["input_tokens"] for x in all_metrics)
        del source_model; torch.cuda.empty_cache()
        if "UNIFORM_RESTART" in args.method:
            pattern = {}
        elif "SHUFFLED_SELF_RESTART" in args.method:
            # This branch is supplied by a self-pilot caller; derive a uniform
            # compatible pattern if no pilot is present.
            pattern = {}
        fresh = load_model(args.model)
        if not pattern:
            names = [n for n, m in fresh.named_modules() if isinstance(m, torch.nn.Linear) and
                     (n.endswith(".q_proj") or n.endswith(".o_proj"))]
            pattern = {n: args.target_rank for n in names}
        fresh = inject_pattern(fresh, pattern, student_init, args.scale, args.method, args.seed + 100000)
        student_steps = args.updates - pilot_steps
        student_updates = updates[pilot_steps:]
        all_metrics += train_updates(fresh, student_updates, train, tokenizer, args.lr,
                                     args.microbatch, student_steps, start_step=pilot_steps)
        model = fresh
        metadata_extra = {"pilot_steps": pilot_steps, "student_steps": student_steps,
                          "pilot_discarded": True, "pilot_pattern": pattern,
                          "pilot_input_tokens": pilot_tokens}
    elif is_compress:
        init = args.init or ("I" if args.method.startswith("I_") else "L")
        wide_steps = max(1, int(round(.75 * args.updates)))
        model = load_model(args.model)
        model = inject(model, args.rank, ("q_proj", "o_proj"), init, args.scale, args.method, args.seed)
        all_metrics += train_updates(model, updates[:wide_steps], train, tokenizer, args.lr,
                                     args.microbatch, wide_steps)
        entries = []
        for name, module in module_pattern(model).items():
            entries.append((name, module, *thin_core_svd(module.A0, module.B0, module.A, module.B, module.scale)))
        if "GLOBAL_SVD" in args.method:
            entries2 = [(n, m, l, s, r, meta) for n, m, l, s, r, meta in entries]
            ranks = {n: args.target_rank for n, *_ in entries2}
            # Select globally largest singular energies with a prefix constraint.
            candidates = sorted(((float(sv * sv), n, j) for n, _, _, s, _, _ in entries2 for j, sv in enumerate(s)), reverse=True)
            ranks = {n: 0 for n, *_ in entries2}; remaining = args.target_rank * len(entries2)
            for _, n, j in candidates:
                if remaining and j == ranks[n]: ranks[n] += 1; remaining -= 1
            factors = export_factors(entries2, ranks)
        else:
            factors = export_factors(entries, {n: args.target_rank for n, *_ in entries})
        del model; torch.cuda.empty_cache()
        model = load_model(args.model)
        pattern = {n: f["rank"] for n, f in factors.items()}
        model = inject_pattern(model, pattern, "L", 1.0, args.method + "_RECOVERY", args.seed + 200000)
        for name, module in module_pattern(model).items():
            with torch.no_grad():
                module.adapter.A.copy_(factors[name]["A"].to(module.A.device))
                module.adapter.B.copy_(factors[name]["B"].to(module.B.device))
        recovery_steps = args.updates - wide_steps
        all_metrics += train_updates(model, updates[wide_steps:], train, tokenizer, args.lr,
                                     args.microbatch, recovery_steps, start_step=wide_steps)
        metadata_extra = {"wide_steps": wide_steps, "recovery_steps": recovery_steps,
                          "compression_policy": "global_svd" if "GLOBAL_SVD" in args.method else "uniform_svd",
                          "export_ranks": pattern,
                          "compression_meta": {n: f["meta"] for n, f in factors.items()}}
    else:
        raise ValueError(f"unsupported pipeline method: {args.method}")
    dev_metrics = evaluate(model, dev, tokenizer, "cuda")
    metrics_path = out / "metrics.json"; metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
    metadata = {"protocol": "adaptive-rank-v2", "method": args.method, "seed": args.seed,
                "scale": args.scale, "lr": args.lr, "updates": args.updates,
                "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "status": "complete", "input_tokens": sum(x["input_tokens"] for x in all_metrics),
                "supervised_tokens": sum(x["supervised_tokens"] for x in all_metrics),
                "elapsed_s": time.time() - t0, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                **dev_metrics, **metadata_extra}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
