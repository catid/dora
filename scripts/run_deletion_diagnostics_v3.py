#!/usr/bin/env python3
"""Shadow deletion diagnostics for corrected L_GATE and I_GATE checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel

from rankstudy.v2.data import collate, read_jsonl
from rankstudy.v3.adapters import adapted, inject
from rankstudy.v3.checkpoint import backbone_hash, json_write, load_adapter
from rankstudy.v3.scoring import token_subsample
from scripts.run_one_v3 import numerator


def adapter_digest(model):
    h = hashlib.sha256()
    for name, module in sorted(adapted(model).items()):
        h.update(name.encode())
        for key, value in sorted(module.state_dict().items()):
            if key.startswith("base."):
                continue
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_digest():
    h = hashlib.sha256(); h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all(): h.update(state.cpu().numpy().tobytes())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def restore_lora_origin(model, run_meta):
    """Recreate A0 for old L checkpoints that did not store it."""
    config = run_meta.get("config", {})
    seed = int(run_meta.get("initialization_seed", config.get("adapter_seed") or config.get("seed", 0)))
    for name, module in adapted(model).items():
        if module.init != "L" or hasattr(module, "A0"):
            continue
        stable = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        generator = torch.Generator().manual_seed((seed + stable) % (2**63))
        origin = torch.empty(module.A.shape, dtype=module.A.dtype, device="cpu").uniform_(
            -1 / math.sqrt(module.base.in_features), 1 / math.sqrt(module.base.in_features), generator=generator)
        module.register_buffer("A0", origin.to(module.A.device))


def load_state(checkpoint, manifest, run_meta, model, device):
    modules = manifest["modules"]
    pattern = {name: entry["rank"] for name, entry in modules.items()}
    first = next(iter(modules.values()))
    inject(model, max(pattern.values()), first["init"], float(first["scale"]), 0,
           first["score"], ("q_proj", "o_proj"), pattern)
    load_adapter(checkpoint, model, device)
    restore_lora_origin(model, run_meta)
    model.eval()
    for module in adapted(model).values(): module.begin_update(False)
    return model


@torch.no_grad()
def loss(model, rows, tokenizer, device, batch_size=8):
    total = 0.0; count = 0; groups = []
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch = collate(batch_rows, tokenizer.pad_token_id, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            for index in range(len(batch_rows)):
                value, tokens = numerator(logits[index:index + 1], batch["labels"][index:index + 1])
                total += float(value); count += int(tokens); groups.append(float(value / tokens))
    return {"token_weighted_nll": total / count, "group_balanced_nll": sum(groups) / len(groups), "groups": groups}


def score_checkpoint(checkpoint, manifest, run_meta, model, rows, tokenizer, device, method, seed):
    model = load_state(checkpoint, manifest, run_meta, model, device)
    # Use one effective score batch so the fixed sample is at most 256 valid
    # tokens total for each module, matching the training runner's contract.
    batches = [collate(rows[:8], tokenizer.pad_token_id, device)]
    for module in adapted(model).values(): module.score = method; module.begin_update(True)
    for ordinal, batch in enumerate(batches, 1):
        sample = token_subsample([batch["attention_mask"]], seed, ordinal, 256)[0].to(device) if method == "ACT" else None
        for module in adapted(model).values(): module._sample_indices = sample
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        value, _ = numerator(logits, batch["labels"]); value.backward()
    supervised = sum(int(batch["labels"].ne(-100).sum()) for batch in batches)
    for parameter in model.parameters():
        if parameter.grad is not None: parameter.grad.div_(supervised)
    scores = {}; signed_gate = {}
    with torch.no_grad():
        for name, module in adapted(model).items():
            scores[name] = ((module._gate.grad / supervised).abs().cpu() if method == "GATE"
                            else module.task_score(1).cpu())
            if module._gate is not None and module._gate.grad is not None:
                signed_gate[name] = (-module._gate.grad / supervised).cpu()
    model.zero_grad(set_to_none=True)
    return model, scores, signed_gate


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i)); result = [0.0] * len(values); pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[pos]]): end += 1
        rank = (pos + end - 1) / 2 + 1
        for index in order[pos:end]: result[index] = rank
        pos = end
    return result


def spearman(left, right):
    if len(left) < 2: return None
    left, right = rankdata(left), rankdata(right); lm, rm = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = (sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)) ** 0.5
    return None if denominator == 0 else numerator / denominator


def stable_slot_probe(active, gate_scores, size, seed):
    """Common probe: equal score quartiles, alternating q_proj/o_proj."""
    if len(active) <= size: return list(active)
    ordered = sorted(active, key=lambda item: (float(gate_scores[item[0]][item[1]]), item[0], item[1]))
    selected = []; per_quartile = max(1, size // 4)
    for quartile in range(4):
        group = ordered[quartile * len(ordered) // 4:(quartile + 1) * len(ordered) // 4]
        buckets = {"q_proj": [], "o_proj": []}
        for item in group: buckets[item[0].split(".")[-1]].append(item)
        for offset, key in enumerate(buckets): random.Random(seed + quartile * 17 + offset).shuffle(buckets[key])
        while len(selected) < (quartile + 1) * per_quartile and any(buckets.values()):
            for key in ("q_proj", "o_proj"):
                if buckets[key] and len(selected) < (quartile + 1) * per_quartile: selected.append(buckets[key].pop())
    if len(selected) < size: selected.extend(item for item in ordered if item not in set(selected))
    return selected[:size]


def delete_and_measure(model, rows, tokenizer, device, slots):
    masks = {name: module.mask.clone() for name, module in adapted(model).items()}
    for name, slot in slots:
        mask = adapted(model)[name].mask.clone(); mask[slot] = False; adapted(model)[name].mask.copy_(mask)
    try: return loss(model, rows, tokenizer, device)
    finally:
        for name, mask in masks.items(): adapted(model)[name].mask.copy_(mask)


def run_one(args, method, progress, checkpoint):
    device = "cuda"; torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest"); torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True); tokenizer.pad_token = tokenizer.eos_token
    train = read_jsonl(args.data_root / "train.jsonl"); indices = json.loads((args.data_root / "probe_source_indices.json").read_text())
    lookup = {row["source_index"]: row for row in train}; rows = [lookup[index] for index in indices]
    manifest = json.loads((checkpoint / "manifest.json").read_text()); run_meta = json.loads((checkpoint.parent / "metadata.json").read_text())
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device)
    reference = load_state(checkpoint, manifest, run_meta, base, device); original_hash = backbone_hash(reference); original_loss = loss(reference, rows, tokenizer, device)
    adapter_before, rng_before = adapter_digest(reference), rng_digest(); scored = {}; gate_signed = {}
    for family in ("RAW", "GATE", "DISP", "ACT"):
        fresh = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device)
        model, scores, signed = score_checkpoint(checkpoint, manifest, run_meta, fresh, rows, tokenizer, device, family, args.seed)
        scored[family] = scores
        if family == "GATE": gate_signed = signed
        del model, fresh
    active = [(name, slot) for name, module in adapted(reference).items() for slot, enabled in enumerate(module.mask.tolist()) if enabled]
    common_probe = stable_slot_probe(active, scored["GATE"], min(64, len(active)), args.seed + int(progress * 100))
    # The actual d_i values do not depend on which estimator is being
    # evaluated. Compute each deletion exactly once and reuse it for all four
    # score-family correlations.
    common_single_after = {
        (name, slot): delete_and_measure(reference, rows, tokenizer, device, [(name, slot)])
        for name, slot in common_probe
    }
    common_group_after = delete_and_measure(reference, rows, tokenizer, device, common_probe)
    records = []
    for family, scores in scored.items():
        ordered = sorted(active, key=lambda item: (-float(scores[item[0]][item[1]]), item[0], item[1]))
        groups = {"score_top64": ordered[:64], "score_bottom64": ordered[-64:], "common_stratified64": common_probe}; group_damage = []
        for label, slots in groups.items():
            after = common_group_after if label == "common_stratified64" else delete_and_measure(reference, rows, tokenizer, device, slots)
            group_damage.append({"selection": label, "token_nll_damage": after["token_weighted_nll"] - original_loss["token_weighted_nll"], "group_nll_damage": after["group_balanced_nll"] - original_loss["group_balanced_nll"], "after": after})
        probes = []
        for name, slot in common_probe:
            after = common_single_after[(name, slot)]
            probes.append({"module": name, "slot": slot, "score": float(scores[name][slot]), "deletion_token_nll": after["token_weighted_nll"] - original_loss["token_weighted_nll"], "deletion_group_nll": after["group_balanced_nll"] - original_loss["group_balanced_nll"], "gate_first_order_prediction": float(gate_signed[name][slot]) if name in gate_signed else None})
        values = [item["score"] for item in probes]; damages = [item["deletion_token_nll"] for item in probes]
        order = sorted(range(len(probes)), key=lambda i: (probes[i]["score"], i)); quartiles = []
        for quartile in range(4):
            ids = order[quartile * len(order) // 4:(quartile + 1) * len(order) // 4]
            quartiles.append({"quartile": quartile, "n": len(ids), "mean_score": sum(probes[i]["score"] for i in ids) / len(ids), "mean_deletion_damage": sum(probes[i]["deletion_token_nll"] for i in ids) / len(ids)})
        records.append({"family": family, "slots_available": len(active), "probe_slots": probes, "rank_correlation_score_vs_damage": spearman(values, damages), "rank_correlation_abs_score_vs_abs_damage": spearman([abs(x) for x in values], [abs(x) for x in damages]), "calibration_quartiles": quartiles, "groups": group_damage, "score_values_summary": {"min": min(values), "max": max(values)}})
    if adapter_digest(reference) != adapter_before or rng_digest() != rng_before: raise RuntimeError("diagnostic failed to restore adapter tensors or RNG state")
    result = {"status": "passed", "method": method, "checkpoint": str(checkpoint), "progress": progress, "diagnostic_seed": args.seed, "rows": len(rows), "active_slots": len(active), "original_backbone_sha256": original_hash, "probe_manifest_sha256": hashlib.sha256((args.data_root / "probe_source_indices.json").read_bytes()).hexdigest(), "original_loss": original_loss, "families": records, "selection_eligible": False, "common_probe_slots": common_probe, "restoration_verified": True, "score_probe_examples": 8, "score_probe_sampling": f"deterministic up to 256 valid tokens per score batch; seed {args.seed}", "limitations": ["Training-side shadow diagnostic only; no dev/test selection.", "Group deletion is a diagnostic, not an oracle allocator."]}
    output = args.out / f"{method}_p{progress}_s{args.seed}"; output.mkdir(parents=True, exist_ok=True); json_write(output / "metrics.json", result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--method", choices=["L_GATE", "I_GATE"], required=True); parser.add_argument("--checkpoint-root", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B"); parser.add_argument("--data-root", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data_v2")); parser.add_argument("--progress", type=float, choices=[.20, .50, .75]); parser.add_argument("--seed", type=int, choices=[2901, 2902], required=True); args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    results = []
    # The replay root contains exact token-progress checkpoints.  The legacy
    # 128/320/448 checkpoints are accepted only as an explicit fallback for
    # older engineering receipts and are labeled by their actual fraction.
    schedule = ((0.20, 103), (0.50, 257), (0.75, 385))
    if args.progress is not None: schedule = tuple(item for item in schedule if item[0] == args.progress)
    for progress, step in schedule:
        root = args.checkpoint_root / args.method
        checkpoint = root / f"p{int(progress * 100)}" / f"checkpoint_{step:04d}"
        if not checkpoint.exists():
            checkpoint = root / f"checkpoint_{step:04d}"
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        results.append(run_one(args, args.method, progress, checkpoint))
    json_write(args.out / f"{args.method}_s{args.seed}_summary.json", {"status": "complete", "method": args.method, "diagnostic_seed": args.seed, "results": results, "selection_eligible": False}); print(json.dumps({"method": args.method, "seed": args.seed, "progress_points": len(results)}))


if __name__ == "__main__": main()
