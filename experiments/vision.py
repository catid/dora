"""Flowers102 transfer benchmark with one fixed pretrained ViT and matched PEFT runs.

Example (GPU 0 only):
  CUDA_VISIBLE_DEVICES=0 python -m experiments.vision --seeds 42 43 44

This trains on the official 1,020-image train split, selects the best epoch on
1,020 validation images, and evaluates each selected checkpoint on 6,149 test
images once. No hyperparameters are selected using the test set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.transforms import InterpolationMode
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from experiments.adapters import AdapterLinear, inject_adapters

MODEL = "vit_base_patch16_224.augreg_in21k_ft_in1k"
REPO = "timm/" + MODEL
REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"
TARGET_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_datasets(cache, data_config):
    mean, std = data_config["mean"], data_config["std"]
    interpolation = InterpolationMode.BICUBIC
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    datasets = {
        split: Flowers102(cache, split=split, download=True,
                          transform=training_transform if split == "train" else evaluation_transform)
        for split in ("train", "val", "test")
    }
    assert {key: len(value) for key, value in datasets.items()} == {
        "train": 1020, "val": 1020, "test": 6149,
    }
    manifests = {
        split: [{"image": str(Path(path).relative_to(cache)), "label": int(label)}
                for path, label in zip(dataset._image_files, dataset._labels)]
        for split, dataset in datasets.items()
    }
    id_sets = [{entry["image"] for entry in manifests[split]} for split in datasets]
    assert all(not id_sets[i].intersection(id_sets[j])
               for i in range(3) for j in range(i + 1, 3))
    return datasets, manifests


def make_loaders(datasets, seed, batch_size, workers):
    return {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0, worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(seed + index * 1000),
                          drop_last=False)
        for index, (split, dataset) in enumerate(datasets.items())
    }


@torch.inference_mode()
def evaluate(model, loader, save_predictions=False):
    model.eval()
    count, correct, loss_sum = 0, 0, 0.0
    class_counts = torch.zeros(102, device="cuda", dtype=torch.int64)
    class_correct = torch.zeros_like(class_counts)
    all_labels, all_predictions = [], []
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        predictions = logits.argmax(1)
        loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
        correct += predictions.eq(labels).sum().item()
        class_counts += torch.bincount(labels, minlength=102)
        class_correct += torch.bincount(labels[predictions.eq(labels)], minlength=102)
        count += labels.numel()
        if save_predictions:
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    result = {"count": count, "correct": correct, "accuracy": correct / count,
              "macro_class_accuracy": (class_correct / class_counts.clamp_min(1)).mean().item(),
              "cross_entropy": loss_sum / count}
    if save_predictions:
        result["labels"], result["predictions"] = all_labels, all_predictions
    return result


def train_epoch(model, loader, optimizer):
    model.train()
    count, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels, label_smoothing=0.1)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        count += labels.numel()
        correct += logits.argmax(1).eq(labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return {"count": count, "accuracy": correct / count,
            "label_smoothed_cross_entropy": loss_sum / count}


def row_norm_diagnostics(model):
    values = []
    for name, module in model.named_modules():
        if not isinstance(module, AdapterLinear):
            continue
        if hasattr(module, "effective_weight"):
            adapted = module.effective_weight().detach().float()
        elif hasattr(module, "to_linear"):
            adapted = module.to_linear().weight.detach().float()
        else:
            return {"available": False, "reason": "No effective-weight accessor"}
        base = module.weight.detach().float()
        reference_norm = base.norm(dim=1)
        ratio = adapted.norm(dim=1) / reference_norm.clamp_min(1e-12)
        entry = {"name": name, "row_norm_relative_change_max": (ratio - 1).abs().max().item(),
                 "row_norm_ratio_mean": ratio.mean().item(),
                 "up_factor_l2_norm": module.lora_B.detach().norm().item()}
        assert entry["up_factor_l2_norm"] > 0, f"Adapter did not learn: {name}"
        if module.use_nora:
            error = (module.effective_down().detach().norm(dim=0) - 1).abs().max().item()
            entry["nora_down_column_norm_max_error"] = error
            assert error < 1e-5
        if module.use_dora:
            magnitude = module.m.detach().flatten().abs()
            error = ((adapted.norm(dim=1) - magnitude).abs() / magnitude.clamp_min(1e-12)).max().item()
            entry["dora_magnitude_relative_max_error"] = error
            assert error < 1e-4
        values.append(entry)
    return {"available": True, "layers": values,
            "max_relative_change": max((v["row_norm_relative_change_max"] for v in values), default=0.0)}


def run_one(args, method, seed, checkpoint, checkpoint_hash, datasets, provenance, evaluate_test=True):
    directory = args.output / f"seed_{seed}" / method
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists() and not args.overwrite:
        existing = json.loads(result_path.read_text())
        expected = {"method": method, "seed": seed, "epochs": args.epochs, "rank": args.rank,
                    "batch_size": args.batch_size, "adapter_learning_rate": args.lr,
                    "head_learning_rate": args.head_lr, "weight_decay": args.weight_decay,
                    "checkpoint_sha256": checkpoint_hash, "evaluate_test": evaluate_test}
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Existing result has different {key}: {result_path}")
        if existing["provenance"]["adapters_sha256"] != provenance["adapters_sha256"]:
            raise RuntimeError(f"Existing result used different adapter implementation: {result_path}")
        print(f"Existing completed result: {result_path}", flush=True)
        return existing
    seed_everything(seed)
    model = timm.create_model(MODEL, pretrained=False)
    source_state = load_file(checkpoint)
    model.load_state_dict(source_state, strict=True)
    # This digest is measured before any head reset or adapter conversion.
    loaded_checkpoint_digest = tensor_digest(model.state_dict().items())
    source_digest = tensor_digest(source_state.items())
    assert loaded_checkpoint_digest == source_digest
    del source_state
    model.reset_classifier(102)
    initial_head_digest = tensor_digest(model.head.state_dict().items())
    initial_backbone_digest = tensor_digest((name, parameter) for name, parameter in model.named_parameters()
                                            if not name.startswith("head."))
    model.eval()
    # Use an isolated generator so identity validation does not alter training RNG.
    identity_input = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(999))
    with torch.no_grad():
        before = model(identity_input)
    if method == "baseline":
        model.requires_grad_(False)
    else:
        inject_adapters(model, method, rank=args.rank,
                        targets=lambda name, module: name.endswith(TARGET_SUFFIXES))
    model.head.requires_grad_(True)
    model.eval()
    with torch.no_grad():
        after = model(identity_input)
    identity_error = (after - before).abs().max().item()
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)
    del identity_input, before, after
    frozen_before = tensor_digest((name, parameter) for name, parameter in model.named_parameters()
                                   if not parameter.requires_grad)
    targeted = [name for name, module in model.named_modules() if isinstance(module, AdapterLinear)]
    assert len(targeted) == (0 if method == "baseline" else 48)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_names = {name for name, _ in trainable}
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    initial_trainable_digest = tensor_digest(trainable)
    model.cuda()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = [{"params": [parameter for name, parameter in trainable if name.startswith("head.")],
               "lr": args.head_lr}]
    adapters = [parameter for name, parameter in trainable if not name.startswith("head.")]
    if adapters:
        groups.append({"params": adapters, "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    def lr_factor(epoch):
        if epoch < 2:
            return (epoch + 1) / 2
        return 0.5 * (1 + math.cos(math.pi * (epoch - 2) / max(1, args.epochs - 2)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    loaders = make_loaders(datasets, seed, args.batch_size, args.workers)
    config = {
        "method": method, "seed": seed, "epochs": args.epochs, "rank": args.rank,
        "batch_size": args.batch_size, "adapter_learning_rate": args.lr,
        "head_learning_rate": args.head_lr, "evaluate_test": evaluate_test,
        "weight_decay": args.weight_decay, "label_smoothing": 0.1,
        "optimizer": "AdamW", "schedule": "2-epoch warmup then cosine",
        "gradient_clip_norm": 1.0, "autocast": "bfloat16", "base_dtype": "float32",
        "targeted_layers": targeted, "trainable_parameters": trainable_count,
        "trainable_names": sorted(trainable_names),
        "initial_head_sha256": initial_head_digest,
        "initial_trainable_sha256": initial_trainable_digest,
        "initial_backbone_sha256": initial_backbone_digest,
        "source_state_sha256": source_digest,
        "loaded_source_state_sha256": loaded_checkpoint_digest,
        "checkpoint_sha256": checkpoint_hash,
        "initial_output_max_absolute_error": identity_error,
        "selection": "maximum validation accuracy, tie-break lower validation cross-entropy",
        "provenance": provenance,
    }
    write_json(directory / "config.json", config)
    print(json.dumps({"event": "start", "method": method, "seed": seed,
                      "trainable_parameters": trainable_count, "initial_identity_error": identity_error}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    best_key, best_epoch, best_state = (-1.0, -math.inf), None, None
    history = []
    with (directory / "epochs.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            training = train_epoch(model, loaders["train"], optimizer)
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - epoch_start
            validation = evaluate(model, loaders["val"])
            torch.cuda.synchronize()
            key = (validation["accuracy"], -validation["cross_entropy"])
            if key > best_key:
                best_key, best_epoch = key, epoch + 1
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items() if name in trainable_names}
                torch.save(best_state, directory / "best_adapter_and_head.pt")
            entry = {"epoch": epoch + 1, "learning_rates": learning_rates,
                     "train": training, "validation": validation,
                     "training_seconds": training_seconds,
                     "epoch_seconds": time.perf_counter() - epoch_start,
                     "best_epoch": best_epoch}
            history.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps({"event": "epoch", "method": method, "seed": seed, **entry}), flush=True)
            scheduler.step()
    training_wall_seconds = time.perf_counter() - start
    assert best_state is not None
    incompatible = model.load_state_dict(best_state, strict=False)
    assert not incompatible.unexpected_keys
    assert all(name not in trainable_names for name in incompatible.missing_keys)
    frozen_after = tensor_digest((name, parameter) for name, parameter in model.named_parameters()
                                  if not parameter.requires_grad)
    assert frozen_before == frozen_after, "Frozen backbone changed during training"
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    final_trainable_digest = tensor_digest((name, parameter) for name, parameter in model.named_parameters()
                                          if parameter.requires_grad)
    assert initial_trainable_digest != final_trainable_digest, "Trainable parameters did not change"
    norms = row_norm_diagnostics(model)
    test, test_seconds = None, 0.0
    if evaluate_test:
        test_start = time.perf_counter()
        test = evaluate(model, loaders["test"], save_predictions=True)
        torch.cuda.synchronize()
        test_seconds = time.perf_counter() - test_start
        write_json(directory / "test_predictions.json", {"labels": test.pop("labels"),
                                                           "predictions": test.pop("predictions")})
    result = {**config, "best_epoch": best_epoch,
              "best_validation": history[best_epoch - 1]["validation"], "test": test,
              "training_wall_seconds": training_wall_seconds, "test_seconds": test_seconds,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "frozen_state_before_sha256": frozen_before,
              "frozen_state_after_sha256": frozen_after,
              "frozen_base_unchanged": frozen_before == frozen_after,
              "final_trainable_sha256": final_trainable_digest,
              "row_norm_diagnostics": norms,
              "best_adapter_checkpoint_sha256": file_sha256(directory / "best_adapter_and_head.pt")}
    write_json(result_path, result)
    print(json.dumps({"event": "result", "method": method, "seed": seed,
                      "test": test, "best_epoch": best_epoch,
                      "training_wall_seconds": training_wall_seconds}), flush=True)
    del model, optimizer, best_state, loaders
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/dora-bench/cache"))
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/runs/vision"))
    parser.add_argument("--methods", nargs="+", default=["baseline", "lora", "dora", "nora", "dora_nora"],
                        choices=["baseline", "lora", "dora", "nora", "dora_nora"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--tune-lrs", nargs="+", type=float, default=[])
    parser.add_argument("--pilot-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The real pretrained benchmark requires a CUDA device")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = hf_hub_download(REPO, "model.safetensors", revision=REVISION,
                                cache_dir=str(args.cache / "huggingface"))
    config_path = hf_hub_download(REPO, "config.json", revision=REVISION,
                                 cache_dir=str(args.cache / "huggingface"))
    checkpoint_hash = file_sha256(checkpoint)
    source_config = json.loads(Path(config_path).read_text())
    data_config = source_config["pretrained_cfg"]
    datasets, manifests = make_datasets(args.cache, data_config)
    write_json(args.output / "split_manifest.json", manifests)
    data_hashes = {str(path.relative_to(args.cache)): file_sha256(path)
                   for path in (args.cache / "flowers-102").glob("*") if path.is_file()}
    provenance = {
        "model": MODEL, "model_hf_repository": REPO, "model_revision": REVISION,
        "model_checkpoint_path": str(checkpoint), "model_checkpoint_sha256": checkpoint_hash,
        "dataset_source": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
        "dataset_split": "official setid.mat train1020/val1020/test6149",
        "dataset_file_sha256": data_hashes,
        "split_manifest_sha256": file_sha256(args.output / "split_manifest.json"),
        "input_transform": {"train": "RandomResizedCrop224 scale[0.7,1.0], horizontalflip0.5",
                            "eval": "Resize256 bicubic, CenterCrop224", "mean": data_config["mean"],
                            "std": data_config["std"]},
        "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
        "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": file_sha256(__file__),
        "adapters_sha256": file_sha256(Path(__file__).with_name("adapters.py")),
        "command": [sys.executable, *sys.argv], "tf32": True,
    }
    write_json(args.output / "provenance.json", provenance)
    results = []
    # Complete the head-only baseline first to expose an early real measurement.
    if "baseline" in args.methods:
        results.append(run_one(args, "baseline", args.seeds[0], checkpoint, checkpoint_hash, datasets, provenance))
        write_json(args.output / "results.json", results)
    selected_lrs = {method: args.lr for method in args.methods}
    if args.tune_lrs:
        pilot_results = []
        for method in args.methods:
            if method == "baseline":
                continue
            trials = []
            for learning_rate in args.tune_lrs:
                pilot_args = copy.copy(args)
                pilot_args.output = args.output / "pilots" / f"lr_{learning_rate:g}"
                pilot_args.epochs = args.pilot_epochs
                pilot_args.lr = learning_rate
                trial = run_one(pilot_args, method, args.seeds[0], checkpoint, checkpoint_hash,
                                datasets, provenance, evaluate_test=False)
                trials.append(trial)
                pilot_results.append(trial)
                write_json(args.output / "pilot_results.json", pilot_results)
            winner = max(trials, key=lambda result: (result["best_validation"]["accuracy"],
                                                     -result["best_validation"]["cross_entropy"]))
            selected_lrs[method] = winner["adapter_learning_rate"]
            write_json(args.output / "selected_learning_rates.json", selected_lrs)
    for seed in args.seeds:
        for method in args.methods:
            if method == "baseline" and seed == args.seeds[0]:
                continue
            run_args = copy.copy(args)
            run_args.lr = selected_lrs[method]
            results.append(run_one(run_args, method, seed, checkpoint, checkpoint_hash, datasets, provenance))
            write_json(args.output / "results.json", results)


if __name__ == "__main__":
    main()
