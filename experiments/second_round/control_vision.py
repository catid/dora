"""Fixed-image, zero-update numerical control for the Aircraft ViT benchmark.

Uses a trained baseline head and fixed validation images. No training,
test-set evaluation, hyperparameter selection, or timing sweep is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from experiments.second_round.vision import datasets_and_manifest
from experiments.second_round_adapters import AdapterLinear, METHODS, inject_adapters
from experiments.vision import MODEL, REPO, REVISION, TARGET_SUFFIXES, file_sha256, tensor_digest, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/var/tmp/dora-bench/round2/vision/rank2_run"))
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/dora-bench/cache/aircraft"))
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/round2/vision/numerical_control"))
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mechanism", action="store_true", help="Save a separate 128-image diagnostic with GPU-initialized DoRA magnitudes")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = True
    protocol = json.loads((args.root / "protocol.json").read_text())
    baseline_record = json.loads((args.root / "baseline/seed_42/result.json").read_text())
    for source, expected in protocol["provenance"]["source_sha256"].items():
        assert file_sha256(source) == expected, source
    head_path = args.root / "baseline/seed_42/best_adapter_and_head.pt"
    assert file_sha256(head_path) == baseline_record["best_adapter_checkpoint_sha256"]
    dataset_args = argparse.Namespace(cache=args.cache, output=args.root)
    datasets, manifest = datasets_and_manifest(dataset_args)
    assert file_sha256(args.root / "split_manifest.json") == protocol["provenance"]["split_manifest_sha256"]
    assert 1 <= args.count <= len(datasets["val"]) and args.batch_size > 0
    batches = list(DataLoader(Subset(datasets["val"], range(args.count)), batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True))
    images = torch.cat([batch[0] for batch in batches])
    labels = torch.cat([batch[1] for batch in batches])
    del batches
    image_hash = tensor_digest([("images", images)])
    images = images.cuda()
    checkpoint = hf_hub_download(REPO, "model.safetensors", revision=REVISION,
                                cache_dir="/var/tmp/dora-bench/cache/huggingface")
    assert file_sha256(checkpoint) == protocol["provenance"]["checkpoint_sha256"]
    seed_everything(42)
    base = timm.create_model(MODEL, pretrained=False)
    base.load_state_dict(load_file(checkpoint), strict=True)
    base.reset_classifier(100)
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    assert set(head) == {"head.weight", "head.bias"}
    restored = base.load_state_dict(head, strict=False)
    assert not restored.unexpected_keys and all(not name.startswith("head.") for name in restored.missing_keys)
    assert tensor_digest((name, parameter) for name, parameter in base.named_parameters() if name.startswith("head.")) == baseline_record["final_trainable_sha256"]
    base.eval().requires_grad_(False)
    base_hash = tensor_digest((name, parameter) for name, parameter in base.named_parameters() if not name.startswith("head."))
    assert base_hash == baseline_record["initial_backbone_sha256"]
    logits, rows, magnitude_errors = {}, [], []
    cases = [("bare", 0), *((method, rank) for rank in (2, 8) for method in METHODS)]
    for method, rank in cases:
        seed_everything(42)
        model = copy.deepcopy(base)
        if method != "bare":
            inject_adapters(model, method, rank=rank, targets=lambda name, module: name.endswith(TARGET_SUFFIXES))
            modules = [module for module in model.modules() if isinstance(module, AdapterLinear)]
            assert len(modules) == 48 and all(torch.count_nonzero(module.lora_B) == 0 for module in modules)
        model.eval().cuda()
        per_module_errors = {}
        for name, module in model.named_modules():
            if isinstance(module, AdapterLinear) and module.use_dora:
                per_module_errors[name] = (module.m.detach() / module._weight_norm() - 1).abs().max().item()
        magnitude_errors.append({"method": method, "rank": rank,
                                 "max_absolute_ratio_error": max(per_module_errors.values(), default=0.0),
                                 "by_module": per_module_errors})
        for precision in ("fp32", "bf16"):
            # FP32 is an IEEE reference; BF16 matches the configured training autocast.
            torch.backends.cuda.matmul.allow_tf32 = precision == "bf16"
            torch.backends.cudnn.allow_tf32 = precision == "bf16"
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = torch.cat([model(batch).float().cpu() for batch in images.split(args.batch_size)])
            assert torch.isfinite(output).all()
            name = f"{method}_r{rank}_{precision}"
            logits[name] = output.numpy()
            reference = logits[f"bare_r0_{precision}"]
            difference = logits[name].astype(np.float64) - reference
            rows.append({"method": method, "rank": rank, "precision": precision,
                         "max_absolute_logit_difference_from_bare": float(np.abs(difference).max()),
                         "mean_absolute_logit_difference_from_bare": float(np.abs(difference).mean()),
                         "rms_logit_difference_from_bare": float(np.sqrt(np.mean(difference ** 2))),
                         "argmax_disagreement_count_from_bare": int(np.sum(logits[name].argmax(1) != reference.argmax(1))),
                         "probe_correct": int(np.sum(logits[name].argmax(1) == labels.numpy()))})
        del model
        torch.cuda.empty_cache()
    pairwise = []
    for precision in ("fp32", "bf16"):
        names = [name for name in logits if name.endswith(precision) and not name.startswith("bare")]
        for first_index, first in enumerate(names):
            for second in names[first_index + 1:]:
                difference = logits[first].astype(np.float64) - logits[second]
                pairwise.append({"first": first, "second": second,
                                 "bitwise_equal": bool(np.array_equal(logits[first].view(np.uint32), logits[second].view(np.uint32))),
                                 "max_absolute_logit_difference": float(np.abs(difference).max()),
                                 "mean_absolute_logit_difference": float(np.abs(difference).mean()),
                                 "argmax_disagreement_count": int(np.sum(logits[first].argmax(1) != logits[second].argmax(1)))})
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "logits.npz", labels=labels.numpy(), **logits)
    result = {"status": "measured", "command": [sys.executable, *sys.argv],
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "timm": timm.__version__, "cuda": torch.version.cuda,
              "source_sha256": file_sha256(__file__),
              "training_source_sha256": protocol["provenance"]["source_sha256"],
              "checkpoint_sha256": protocol["provenance"]["checkpoint_sha256"],
              "split_manifest_sha256": protocol["provenance"]["split_manifest_sha256"],
              "baseline_head_checkpoint_sha256": file_sha256(head_path),
              "baseline_head_tensor_sha256": baseline_record["final_trainable_sha256"],
              "baseline_backbone_tensor_sha256": base_hash,
              "split": "validation", "count": len(labels), "batch_size": args.batch_size, "ranks": [2, 8], "seed": 42,
              "precision": {"fp32": "FP32 base/adapters, no autocast, TF32 matrix multiplication and convolutions disabled",
                            "bf16": protocol["provenance"]["precision"]},
              "image_ids": [row["image"] for row in manifest["splits"]["val"][:args.count]],
              "preprocessed_images_tensor_sha256": image_hash, "logits_sha256": file_sha256(args.output / "logits.npz"),
              "rows": rows, "pairwise_adapter_comparisons": pairwise,
              "initial_cpu_magnitude_vs_gpu_norm": magnitude_errors,
              "all_adapters_bitwise_equal_by_precision": {precision: all(row["bitwise_equal"] for row in pairwise if row["first"].endswith(precision)) for precision in ("fp32", "bf16")},
              "scope": "Fixed validation images in manifest order with trained baseline seed-42 head and zero up factors. Quantifies numerical initialization differences; does not evaluate trained adapters, estimate held-out method performance, or choose settings."}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "control_vision.py").write_bytes(Path(__file__).read_bytes())
    if args.mechanism:
        probe_count = min(128, len(labels))
        seed_everything(42)
        calibrated = copy.deepcopy(base)
        inject_adapters(calibrated, "dora", rank=8, targets=lambda name, module: name.endswith(TARGET_SUFFIXES))
        calibrated.eval().cuda()
        before_errors, after_errors = {}, {}
        with torch.no_grad():
            for name, module in calibrated.named_modules():
                if isinstance(module, AdapterLinear):
                    norm = module._weight_norm()
                    before_errors[name] = (module.m / norm - 1).abs().max().item()
                    module.m.copy_(norm)
                    after_errors[name] = (module.m / module._weight_norm() - 1).abs().max().item()
        mechanism_logits = {"labels": labels[:probe_count].numpy()}
        comparisons = []
        for precision in ("fp32", "bf16"):
            torch.backends.cuda.matmul.allow_tf32 = precision == "bf16"
            torch.backends.cudnn.allow_tf32 = precision == "bf16"
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = calibrated(images[:probe_count]).float().cpu().numpy()
            mechanism_logits[f"dora_gpu_m_{precision}"] = output
            for reference in ("bare_r0", "lora_r8", "dora_r8"):
                reference_name = f"{reference}_{precision}"
                expected = logits[reference_name][:probe_count]
                mechanism_logits[reference_name] = expected
                delta = output.astype(np.float64) - expected
                comparisons.append({"precision": precision, "reference": reference_name,
                                    "bitwise_equal": bool(np.array_equal(output.view(np.uint32), expected.view(np.uint32))),
                                    "max_absolute_logit_difference": float(np.abs(delta).max()),
                                    "mean_absolute_logit_difference": float(np.abs(delta).mean()),
                                    "argmax_disagreement_count": int(np.sum(output.argmax(1) != expected.argmax(1)))})
        np.savez_compressed(args.output / "mechanism_logits.npz", **mechanism_logits)
        mechanism = {"count": probe_count, "source_control_result_sha256": file_sha256(args.output / "result.json"),
                     "logits_sha256": file_sha256(args.output / "mechanism_logits.npz"),
                     "initial_ratio_error_max": max(before_errors.values()),
                     "recalibrated_ratio_error_max": max(after_errors.values()),
                     "initial_ratio_error_by_module": before_errors, "recalibrated_ratio_error_by_module": after_errors,
                     "comparisons": comparisons,
                     "scope": "Separate newly constructed zero-update DoRA rank-8 instance; replace its magnitudes with GPU weight norms before inference on the first 128 validation images. No trained adapter or original control case is modified."}
        (args.output / "mechanism.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    print(json.dumps({"rows": rows, "all_adapters_bitwise_equal_by_precision": result["all_adapters_bitwise_equal_by_precision"]}), flush=True)


if __name__ == "__main__":
    main()
