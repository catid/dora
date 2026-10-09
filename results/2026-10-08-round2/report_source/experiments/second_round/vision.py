"""Harder transfer benchmark: official FGVC-Aircraft variant classification.

The first command measures head-only validation headroom and throughput without
scoring test images. Freeze final budgets before running the second command::

    CUDA_VISIBLE_DEVICES=0 python -m experiments.second_round.vision --phase baseline
    CUDA_VISIBLE_DEVICES=0 python -m experiments.second_round.vision --phase run --epochs 20

All six methods receive four equal validation-only trials per rank. Ranks 2 and
8 remain separate results; held-out test scores never choose settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import FGVCAircraft
from torchvision.transforms import InterpolationMode
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from experiments.vision import (MODEL, REPO, REVISION, TARGET_SUFFIXES, file_sha256,
                                tensor_digest, seed_everything, seed_worker, write_json)
from experiments.second_round_adapters import AdapterLinear, METHODS, inject_adapters, parameter_groups

BASE_LRS = (1e-5, 1e-4, 3e-4, 1e-3)
MLR_BASE_LRS = (1e-4, 1e-3)
MAGNITUDE_MULTIPLIERS = (0.1, 0.01)


def aircraft_loader(path):
    # The dataset's bottom 20-pixel copyright band is not an aircraft feature.
    with Image.open(path) as original:
        image = original.convert("RGB")
    return image.crop((0, 0, image.width, image.height - 20))


def datasets_and_manifest(args):
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    training = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    evaluation = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), normalize,
    ])
    datasets = {split: FGVCAircraft(args.cache, split=split, annotation_level="variant", download=False,
                                   loader=aircraft_loader, transform=training if split == "train" else evaluation)
                for split in ("train", "val", "test")}
    assert {key: len(value) for key, value in datasets.items()} == {"train": 3334, "val": 3333, "test": 3333}
    assert len(datasets["train"].classes) == 100
    assert all(dataset.classes == datasets["train"].classes for dataset in datasets.values())
    manifest_path = args.output / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for split, dataset in datasets.items():
            expected = manifest["splits"][split]
            lookup = {str(Path(path).relative_to(args.cache)): (path, label)
                      for path, label in zip(dataset._image_files, dataset._labels)}
            dataset._image_files = [lookup[row["image"]][0] for row in expected]
            dataset._labels = [lookup[row["image"]][1] for row in expected]
            assert all(lookup[row["image"]][1] == row["label"] for row in expected)
        return datasets, manifest
    manifest = {"classes": datasets["train"].classes,
                "official_sizes": {split: len(dataset) for split, dataset in datasets.items()},
                "splits": {}, "excluded_exact_duplicate_images": {},
                "duplicate_policy": "SHA256 of original JPEG bytes, priority test then validation then train",
                "image_split_limitation": "Official image splits; aircraft registration/photographer grouping is unavailable."}
    seen_paths, seen_hashes = set(), set()
    for split in ("test", "val", "train"):
        dataset = datasets[split]
        keep_paths, keep_labels, rows = [], [], []
        excluded = 0
        for path, label in zip(dataset._image_files, dataset._labels):
            relative = str(Path(path).relative_to(args.cache))
            assert relative not in seen_paths
            seen_paths.add(relative)
            digest = file_sha256(path)
            if digest in seen_hashes:
                excluded += 1
                continue
            seen_hashes.add(digest)
            keep_paths.append(path); keep_labels.append(label)
            rows.append({"image": relative, "label": int(label), "sha256": digest})
        dataset._image_files, dataset._labels = keep_paths, keep_labels
        manifest["splits"][split] = rows
        manifest["excluded_exact_duplicate_images"][split] = excluded
    manifest["effective_sizes"] = {split: len(dataset) for split, dataset in datasets.items()}
    write_json(manifest_path, manifest)
    return datasets, manifest


def loaders_for(datasets, args, seed):
    return {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"),
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed + index * 1000))
            for index, (split, dataset) in enumerate(datasets.items())}


@torch.inference_mode()
def evaluate(model, loader, keep_predictions=False):
    model.eval()
    correct = torch.zeros(100, device="cuda", dtype=torch.int64)
    counts = torch.zeros_like(correct)
    loss_sum, total, top5_total = 0.0, 0, 0
    predictions = {key: [] for key in ("labels", "predictions", "true_class_nll", "confidence", "top5_correct")}
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        logits = logits.float()
        nll = F.cross_entropy(logits, labels, reduction="none")
        guesses = logits.argmax(1)
        top5 = logits.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).any(1)
        counts += torch.bincount(labels, minlength=100)
        correct += torch.bincount(labels[guesses.eq(labels)], minlength=100)
        total += labels.numel(); top5_total += top5.sum().item(); loss_sum += nll.sum().item()
        if keep_predictions:
            for key, value in (("labels", labels), ("predictions", guesses), ("true_class_nll", nll),
                               ("confidence", logits.softmax(1).max(1).values), ("top5_correct", top5)):
                predictions[key].extend(value.cpu().tolist())
    result = {"count": total, "correct": correct.sum().item(),
              "accuracy": correct.sum().item() / total,
              "macro_class_accuracy": (correct / counts.clamp_min(1)).mean().item(),
              "top5_accuracy": top5_total / total, "cross_entropy": loss_sum / total}
    return (result, predictions) if keep_predictions else result


def train_epoch(model, loader, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels, label_smoothing=0.1)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        total += len(labels); correct += logits.argmax(1).eq(labels).sum().item()
        loss_sum += loss.item() * len(labels)
    return {"count": total, "accuracy": correct / total, "label_smoothed_cross_entropy": loss_sum / total}


@torch.no_grad()
def adapter_diagnostics(model):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, AdapterLinear):
            continue
        merged = module.to_linear().weight.float()
        row = {"name": name, "up_factor_l2": module.lora_B.norm().item()}
        assert row["up_factor_l2"] > 0
        if module.use_nora:
            expected = module.log_gain.exp() if module.log_gain is not None else torch.ones(module.in_features, device=merged.device)
            relative = (module.effective_down().norm(dim=0) / expected - 1).abs().max().item()
            row["normalized_column_relative_error"] = relative
            assert relative < 1e-5
        if module.use_dora:
            error = (merged.norm(dim=1) / module.m.flatten().abs().clamp_min(1e-12) - 1).abs().max().item()
            row["merged_magnitude_relative_error"] = error
            assert error < 1e-4
        if module.log_gain is not None:
            gain = module.log_gain.exp()
            row["positive_gain_range"] = [gain.min().item(), gain.max().item()]
            assert torch.isfinite(gain).all() and (gain > 0).all()
        rows.append(row)
    return rows


def run_one(args, datasets, checkpoint, provenance, method, rank, seed, lr, multiplier, stage):
    if stage == "pilot":
        directory = args.output / f"rank_{rank}" / "pilots" / method / f"lr_{lr:g}_m_{multiplier:g}"
    elif stage == "baseline_pilot":
        directory = args.output / "baseline_pilot"
    elif method == "baseline":
        directory = args.output / "baseline" / f"seed_{seed}"
    else:
        directory = args.output / f"rank_{rank}" / "final" / method / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    epochs = args.pilot_epochs if stage in ("pilot", "baseline_pilot") else args.epochs
    evaluate_test = stage == "final"
    config = {"method": method, "rank": rank, "seed": seed, "stage": stage, "epochs": epochs,
              "batch_size": args.batch_size, "adapter_learning_rate": lr,
              "magnitude_lr_multiplier": multiplier, "head_learning_rate": args.head_lr,
              "optimizer": "AdamW", "factor_weight_decay": 0.01, "magnitude_and_gain_weight_decay": 0.0,
              "head_weight_decay": 0.01, "label_smoothing": 0.1, "gradient_clip_norm": 1.0,
              "selection": "maximum validation macro-class accuracy, then minimum validation cross-entropy",
              "evaluate_test": evaluate_test, "provenance": provenance}
    result_path = directory / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        for key in ("method", "rank", "seed", "stage", "epochs", "batch_size", "adapter_learning_rate",
                    "magnitude_lr_multiplier", "head_learning_rate", "evaluate_test"):
            assert existing[key] == config[key], (directory, key)
        assert existing["provenance"]["source_sha256"] == provenance["source_sha256"]
        return existing
    seed_everything(seed)
    model = timm.create_model(MODEL, pretrained=False)
    state = load_file(checkpoint)
    source_digest = tensor_digest(state.items())
    model.load_state_dict(state, strict=True)
    assert tensor_digest(model.state_dict().items()) == source_digest
    del state
    model.reset_classifier(100)
    head_digest = tensor_digest(model.head.state_dict().items())
    backbone_digest = tensor_digest((name, parameter) for name, parameter in model.named_parameters()
                                    if not name.startswith("head."))
    model.eval()
    probe = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(999))
    with torch.no_grad(): before = model(probe)
    if method == "baseline":
        model.requires_grad_(False)
    else:
        inject_adapters(model, method, rank=rank, targets=lambda name, module: name.endswith(TARGET_SUFFIXES))
    model.head.requires_grad_(True)
    model.eval()
    with torch.no_grad(): after = model(probe)
    torch.testing.assert_close(after, before, atol=2e-5, rtol=1e-4)
    identity_error = (after - before).abs().max().item()
    del before, after, probe
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    initial_trainable = tensor_digest((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    frozen_before = tensor_digest((name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad)
    assert frozen_before == backbone_digest
    assert sum(isinstance(module, AdapterLinear) for module in model.modules()) == (0 if method == "baseline" else 48)
    model.cuda()
    groups = [] if method == "baseline" else parameter_groups(model, lr, magnitude_lr_multiplier=multiplier)
    head = list(model.head.parameters())
    groups.append({"params": head, "lr": args.head_lr, "weight_decay": 0.01,
                   "group_name": "classifier", "parameter_count": sum(p.numel() for p in head),
                   "param_names": ["head.weight", "head.bias"]})
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    config.update({"trainable_parameters": trainable_count, "trainable_names": sorted(names),
                   "optimizer_groups": [{key: value for key, value in group.items() if key != "params"} for group in groups],
                   "source_state_sha256": source_digest, "initial_head_sha256": head_digest,
                   "initial_backbone_sha256": backbone_digest, "initial_trainable_sha256": initial_trainable,
                   "initial_output_max_absolute_error": identity_error})
    write_json(directory / "config.json", config)
    optimizer = torch.optim.AdamW(groups)
    def schedule(epoch):
        if epoch < 2: return (epoch + 1) / 2
        return 0.5 * (1 + math.cos(math.pi * (epoch - 2) / max(1, epochs - 2)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loaders = loaders_for(datasets, args, seed)
    best_key, best_epoch, best_state = (-1.0, -math.inf), None, None
    print(json.dumps({"event": "start", "stage": stage, "method": method, "rank": rank, "seed": seed,
                      "lr": lr, "magnitude_multiplier": multiplier, "trainable_parameters": trainable_count}), flush=True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    history = []
    with (directory / "epochs.jsonl").open("w") as log:
        for epoch in range(epochs):
            torch.cuda.synchronize(); epoch_start = time.perf_counter()
            training = train_epoch(model, loaders["train"], optimizer)
            torch.cuda.synchronize(); training_seconds = time.perf_counter() - epoch_start
            validation = evaluate(model, loaders["val"])
            key = (validation["macro_class_accuracy"], -validation["cross_entropy"])
            if key > best_key:
                best_key, best_epoch = key, epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}
                torch.save(best_state, directory / "best_adapter_and_head.pt")
            torch.cuda.synchronize()
            entry = {"epoch": epoch + 1, "learning_rates": [group["lr"] for group in optimizer.param_groups],
                     "train": training, "validation": validation, "best_epoch": best_epoch,
                     "training_seconds": training_seconds, "epoch_seconds": time.perf_counter() - epoch_start}
            history.append(entry); log.write(json.dumps(entry) + "\n"); log.flush()
            print(json.dumps({"event": "epoch", "stage": stage, "method": method, "rank": rank, "seed": seed, **entry}), flush=True)
            scheduler.step()
    duration = time.perf_counter() - started
    restored = model.load_state_dict(best_state, strict=False)
    assert not restored.unexpected_keys and all(name not in names for name in restored.missing_keys)
    frozen_after = tensor_digest((name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad)
    assert frozen_after == frozen_before and all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    final_trainable = tensor_digest((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert initial_trainable != final_trainable
    diagnostics = adapter_diagnostics(model)
    test, test_seconds = None, 0.0
    if evaluate_test:
        test_started = time.perf_counter()
        test, predictions = evaluate(model, loaders["test"], keep_predictions=True)
        torch.cuda.synchronize(); test_seconds = time.perf_counter() - test_started
        write_json(directory / "test_predictions.json", predictions)
    result = {**config, "best_epoch": best_epoch, "best_validation": history[best_epoch - 1]["validation"],
              "test": test, "training_wall_seconds": duration, "test_seconds": test_seconds,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "frozen_base_unchanged": frozen_before == frozen_after,
              "frozen_state_before_sha256": frozen_before, "frozen_state_after_sha256": frozen_after,
              "final_trainable_sha256": final_trainable, "adapter_diagnostics": diagnostics,
              "best_adapter_checkpoint_sha256": file_sha256(directory / "best_adapter_and_head.pt")}
    write_json(result_path, result)
    print(json.dumps({"event": "result", "stage": stage, "method": method, "rank": rank, "seed": seed,
                      "best_validation": result["best_validation"], "test": test,
                      "training_wall_seconds": duration}), flush=True)
    del model, optimizer, loaders, best_state
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "run"), default="baseline")
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/dora-bench/cache/aircraft"))
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/round2/vision"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pilot-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    datasets, manifest = datasets_and_manifest(args)
    checkpoint = hf_hub_download(REPO, "model.safetensors", revision=REVISION,
                                cache_dir="/var/tmp/dora-bench/cache/huggingface")
    sources = ["experiments/second_round/vision.py", "experiments/second_round_adapters.py",
               "experiments/adapters.py", "experiments/vision.py", "dora.py"]
    provenance = {"model": MODEL, "model_repository": REPO, "model_revision": REVISION,
                  "checkpoint_sha256": file_sha256(checkpoint), "dataset": "FGVC-Aircraft variants",
                  "official_split_sizes": manifest["official_sizes"], "split_sizes": manifest["effective_sizes"],
                  "split_manifest_sha256": file_sha256(args.output / "split_manifest.json"),
                  "dataset_archive": json.loads((args.cache / "download_provenance.json").read_text()),
                  "source_sha256": {name: file_sha256(name) for name in sources},
                  "transform": "Remove bottom20 copyright pixels; train RRC224 scale[0.7,1]+horizontalflip; eval resize256+center224; bicubic; mean/std0.5",
                  "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
                  "python": platform.python_version(), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "precision": "FP32 base/adapters, BF16 autocast; TF32 permitted", "command": [sys.executable, *sys.argv]}
    for source in sources:
        destination = args.output / "source" / source; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert file_sha256(destination) == file_sha256(source)
        else: destination.write_bytes(Path(source).read_bytes())
    if args.phase == "baseline":
        run_one(args, datasets, checkpoint, provenance, "baseline", 0, 42, args.head_lr, 1.0, "baseline_pilot")
        return
    protocol = {"epochs": args.epochs, "pilot_epochs": args.pilot_epochs, "ranks": args.ranks,
                "seeds": args.seeds, "batch_size": args.batch_size, "head_lr": args.head_lr,
                "standard_and_gain_lr_grid": BASE_LRS, "mlr_base_lr_grid": MLR_BASE_LRS,
                "magnitude_multipliers": MAGNITUDE_MULTIPLIERS,
                "pilot_seed": 42, "trials_per_method_per_rank": 4,
                "selection": "validation macro accuracy then cross-entropy; no test-driven choices",
                "provenance": provenance}
    path = args.output / "protocol.json"
    if path.exists(): assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else: write_json(path, protocol)
    results = []
    for seed in args.seeds:
        results.append(run_one(args, datasets, checkpoint, provenance, "baseline", 0, seed, args.head_lr, 1.0, "final"))
        write_json(args.output / "results.json", results)
    for rank in args.ranks:
        pilots, selected = [], {}
        for method in METHODS:
            candidates = [(lr, multiplier) for lr in MLR_BASE_LRS for multiplier in MAGNITUDE_MULTIPLIERS] if method == "dora_nora_mlr" else [(lr, 1.0) for lr in BASE_LRS]
            trials = []
            for lr, multiplier in candidates:
                result = run_one(args, datasets, checkpoint, provenance, method, rank, 42, lr, multiplier, "pilot")
                trials.append(result); pilots.append(result)
                write_json(args.output / f"rank_{rank}" / "pilot_results.json", pilots)
            winner = max(trials, key=lambda result: (result["best_validation"]["macro_class_accuracy"], -result["best_validation"]["cross_entropy"]))
            selected[method] = {"lr": winner["adapter_learning_rate"], "magnitude_multiplier": winner["magnitude_lr_multiplier"]}
            write_json(args.output / f"rank_{rank}" / "selected.json", selected)
        for seed in args.seeds:
            for method in METHODS:
                choice = selected[method]
                results.append(run_one(args, datasets, checkpoint, provenance, method, rank, seed,
                                       choice["lr"], choice["magnitude_multiplier"], "final"))
                write_json(args.output / "results.json", results)


if __name__ == "__main__":
    main()
