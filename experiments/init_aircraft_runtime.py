"""Isolated nonidentity Aircraft study runtime.

Copied from frozen second_round.vision.run_one. Training/optimizer/checkpoint/
scoring code is unchanged; only explicit nonidentity/frozen-base policies and
initial validation/probe diagnostics are added. Centered studies continue to
call the original function. Quantized storage is dense, with no speed claim.
"""
from __future__ import annotations
import json
import math
import time
import torch
from safetensors.torch import load_file
import timm
from experiments.second_round.vision import (ADAPTER_TYPES, inject_adapters, parameter_groups,
    adapter_diagnostics, MODEL, TARGET_SUFFIXES, seed_everything, tensor_digest,
    file_sha256, write_json, loaders_for, train_epoch, evaluate)

def run_one(args, datasets, checkpoint, provenance, method, rank, seed, lr, multiplier, stage, backend=None, study_lane="centered"):
    assert study_lane in ("centered", "warmstart", "quantized")
    # Explicit adapter dependency injection for isolated per-change experiments.
    # The existing runner defaults retain the original training/evaluation path.
    adapter_types = ADAPTER_TYPES if backend is None else backend.adapter_types
    inject = inject_adapters if backend is None else backend.inject_adapters
    make_groups = parameter_groups if backend is None else backend.parameter_groups
    diagnose = adapter_diagnostics if backend is None else backend.adapter_diagnostics
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
    reference_initialization = None
    if args.reference_root is not None:
        reference_path = args.reference_root / "baseline" / f"seed_{seed}" / "result.json"
        reference = json.loads(reference_path.read_text())
        assert reference["initial_head_sha256"] == head_digest
        assert reference["initial_backbone_sha256"] == backbone_digest
        assert reference["provenance"]["checkpoint_sha256"] == provenance["checkpoint_sha256"]
        reference_initialization = {"result_path": str(reference_path), "result_sha256": file_sha256(reference_path),
                                    "initial_head_exact": True, "initial_backbone_exact": True}
    model.eval()
    probe = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(999))
    with torch.no_grad(): before = model(probe)
    if method == "baseline":
        model.requires_grad_(False)
    else:
        inject(model, method, rank=rank, targets=lambda name, module: name.endswith(TARGET_SUFFIXES))
    model.head.requires_grad_(True)
    model.eval()
    with torch.no_grad(): after = model(probe)
    if not torch.isfinite(after).all() or not torch.isfinite(before).all():
        raise RuntimeError("Nonfinite initializer output")
    if study_lane == "centered":
        torch.testing.assert_close(after, before, atol=2e-5, rtol=1e-4)
    initial_probe_mse = float((after-before).square().mean())
    initial_probe_relative_l2 = float((after-before).norm()/before.norm())
    identity_error = (after - before).abs().max().item()
    del before, after, probe
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    initial_trainable = tensor_digest((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    frozen_before = tensor_digest((name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad)
    if study_lane != "quantized" or getattr(backend,"initializer_kind",None)!="quantized":
        assert frozen_before == backbone_digest
    else:
        assert frozen_before != backbone_digest
    if backend is not None and hasattr(backend,"verify_initial_frozen"):
        backend.verify_initial_frozen(model,frozen_before)
    assert sum(isinstance(module, adapter_types) for module in model.modules()) == (0 if method == "baseline" else 48)
    model.cuda()
    groups = [] if method == "baseline" else make_groups(model, lr, magnitude_lr_multiplier=multiplier)
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
                   "reference_initialization": reference_initialization,
                   "initial_output_max_absolute_error": identity_error, "initial_probe_mse": initial_probe_mse,
                   "initial_probe_relative_l2": initial_probe_relative_l2, "study_lane":study_lane,
                   "initial_injected_frozen_sha256":frozen_before})
    write_json(directory / "config.json", config)
    optimizer = torch.optim.AdamW(groups)
    def schedule(epoch):
        if epoch < 2: return (epoch + 1) / 2
        return 0.5 * (1 + math.cos(math.pi * (epoch - 2) / max(1, epochs - 2)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loaders = loaders_for(datasets, args, seed)
    initial_validation, initial_predictions = evaluate(model, loaders["val"], keep_predictions=True)
    write_json(directory / "initial_validation_predictions.json", initial_predictions)
    config["initial_validation_predictions_sha256"] = file_sha256(directory / "initial_validation_predictions.json")
    config["initial_validation"] = initial_validation
    write_json(directory / "config.json", config)
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
    diagnostics = diagnose(model)
    test, test_seconds = None, 0.0
    if evaluate_test:
        test_started = time.perf_counter()
        test, predictions = evaluate(model, loaders["test"], keep_predictions=True)
        torch.cuda.synchronize(); test_seconds = time.perf_counter() - test_started
        write_json(directory / "test_predictions.json", predictions)
    if backend is not None:
        diagnostics = diagnose(model)  # Include test-time norm-floor observations.
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

