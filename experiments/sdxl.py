"""Reproducible five-photo SDXL subject-adaptation comparison.

Run only on an explicitly allocated GPU. This is a small subject overfitting
experiment, not a benchmark of general image quality: three training photos,
one validation photo, and one held-out test photo depict the same corgi against
the same orange backdrop. Noise MSE, CLIP alignment, and DINO similarity measure
different things and are reported separately, alongside the actual images.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("HF_HOME", "/var/tmp/dora-bench/huggingface")

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F

from experiments.adapters import AdapterLinear, METHODS, inject_adapters


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
DATASET_ID = "diffusers/dog-example"
DATASET_REVISION = "7aac740a23542bde4c568131b815a9ed08a7e192"
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DINO_ID = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SUBJECT_PROMPT = "a photo of sks dog"
PROMPTS = [SUBJECT_PROMPT, "a photo of sks dog sitting on a beach",
           "a photo of sks dog in a snowy forest", "a watercolor painting of sks dog"]
TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_dataset(output, resolution):
    from huggingface_hub import snapshot_download
    dataset_path = snapshot_download(DATASET_ID, repo_type="dataset", revision=DATASET_REVISION)
    paths = sorted(Path(dataset_path).glob("*.jpeg"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five pinned photos, found {len(paths)}")
    image_dir = output / "prepared_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    canvas = Image.new("RGB", (320 * 5, 365), "white")
    for index, path in enumerate(paths):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        original_size = (image.height, image.width)
        ratio = resolution / min(image.size)
        resized = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS)
        top, left = (resized.height - resolution) // 2, (resized.width - resolution) // 2
        crop = resized.crop((left, top, left + resolution, top + resolution))
        prepared = image_dir / f"{index}.png"
        crop.save(prepared)
        split = "train" if index < 3 else ("validation" if index == 3 else "test")
        records.append({"index": index, "split": split, "source": str(path), "filename": path.name,
                        "sha256": sha256(path), "prepared": str(prepared),
                        "time_ids": [*original_size, top, left, resolution, resolution]})
        canvas.paste(crop.resize((320, 320)), (index * 320, 0))
        ImageDraw.Draw(canvas).text((index * 320 + 8, 330), f"{index}: {split}", fill="black")
    canvas.save(output / "dataset_contact_sheet.jpg")
    write_json(output / "dataset_manifest.json", {"id": DATASET_ID, "revision": DATASET_REVISION,
                                                 "split_rule": "sorted filenames: first 3 train, next validation, last test",
                                                 "records": records})
    return records


def conditioning_metadata(records, resolution):
    return {"model": MODEL_ID, "model_revision": MODEL_REVISION, "weight_variant": "fp16 converted to bf16",
            "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION, "resolution": resolution,
            "prompts": PROMPTS, "latent_method": "posterior mode; fixed center crop",
            "photos": [{key: record[key] for key in ("index", "sha256", "time_ids")} for record in records]}


@torch.no_grad()
def cache_conditioning(pipe, records, output, device, resolution, shared_cache=None):
    cache_path = shared_cache or output / f"conditioning_{resolution}.pt"
    expected = conditioning_metadata(records, resolution)
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists():
        if not metadata_path.exists():
            raise RuntimeError(f"Conditioning cache requires explicit metadata: {metadata_path}")
        saved = json.loads(metadata_path.read_text())
        if saved.get("metadata") != expected or saved.get("sha256") != sha256(cache_path):
            raise RuntimeError("Conditioning cache metadata or SHA256 does not match this experiment")
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    if shared_cache is not None:
        raise FileNotFoundError(f"Requested conditioning cache does not exist: {shared_cache}")
    pipe.vae.to(device=device, dtype=torch.float32)
    latents = []
    for record in records:
        array = np.asarray(Image.open(record["prepared"]), dtype=np.float32) / 127.5 - 1
        image = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
        latent = pipe.vae.encode(image).latent_dist.mode() * pipe.vae.config.scaling_factor
        latents.append(latent.to(dtype=torch.bfloat16).cpu())
    conditionings = []
    for prompt in PROMPTS:
        values = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                    do_classifier_free_guidance=True, negative_prompt="")
        conditionings.append([tensor.cpu() for tensor in values])
    cache = {"latents": torch.cat(latents), "time_ids": torch.tensor([r["time_ids"] for r in records]),
             "conditionings": conditionings, "resolution": resolution}
    torch.save(cache, cache_path)
    write_json(metadata_path, {"metadata": expected, "sha256": sha256(cache_path)})
    return cache


def denoise(unet, noisy, timesteps, conditioning, time_ids):
    return unet(noisy, timesteps, encoder_hidden_states=conditioning[0],
                added_cond_kwargs={"text_embeds": conditioning[2], "time_ids": time_ids}).sample


@torch.no_grad()
def evaluate_mse(unet, scheduler, cache, index, device):
    unet.eval()
    latent = cache["latents"][index:index + 1].to(device)
    conditioning = [t.to(device) for t in cache["conditionings"][0]]
    time_ids = cache["time_ids"][index:index + 1].to(device, dtype=torch.bfloat16)
    rows = []
    start = time.perf_counter()
    for timestep in (100, 300, 500, 700, 900):
        t = torch.tensor([timestep], device=device, dtype=torch.long)
        for seed in (1001, 1002, 1003, 1004):
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
            noisy = scheduler.add_noise(latent, noise, t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = denoise(unet, noisy, t, conditioning, time_ids)
            loss = F.mse_loss(predicted.float(), noise.float()).item()
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite evaluation loss")
            rows.append({"timestep": timestep, "noise_seed": seed, "mse": loss})
    torch.cuda.synchronize(device)
    return {"image_index": index, "mean_mse": float(np.mean([r["mse"] for r in rows])),
            "std_across_noise_timestep": float(np.std([r["mse"] for r in rows])),
            "rows": rows, "seconds": time.perf_counter() - start}


def train(unet, scheduler, cache, device, steps, lr, log_path, seed=42):
    unet.train()
    params = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    latents = cache["latents"][:3].to(device)
    time_ids = cache["time_ids"][:3].to(device, dtype=torch.bfloat16)
    conditioning = [t.to(device) for t in cache["conditionings"][0]]
    schedule = torch.randint(3, (steps,), generator=torch.Generator().manual_seed(2027 + seed - 42)).tolist()
    generator = torch.Generator(device=device).manual_seed(2028 + seed - 42)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = []
    with open(log_path, "w") as stream:
        for step, index in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            latent = latents[index:index + 1]
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
            timesteps = torch.randint(scheduler.config.num_train_timesteps, (1,), generator=generator, device=device)
            noisy = scheduler.add_noise(latent, noise, timesteps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = denoise(unet, noisy, timesteps, conditioning, time_ids[index:index + 1])
                loss = F.mse_loss(predicted.float(), noise.float())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite train value at step {step + 1}")
            optimizer.step()
            value = loss.item()
            losses.append(value)
            record = {"step": step + 1, "image_index": index, "timestep": timesteps.item(),
                      "loss": value, "gradient_norm_before_clip": float(grad_norm),
                      "elapsed_seconds": time.perf_counter() - start}
            stream.write(json.dumps(record) + "\n")
            if (step + 1) % 25 == 0 or step == 0:
                stream.flush()
                print(f"TRAIN {log_path.parent.name} step={step + 1}/{steps} loss={value:.6f} "
                      f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    stats = {"steps": steps, "batch_size": 1, "lr": lr, "seconds": elapsed,
             "steps_per_second": steps / elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
             "first_25_mean_loss": float(np.mean(losses[:25])), "last_25_mean_loss": float(np.mean(losses[-25:]))}
    del optimizer
    return stats


@torch.no_grad()
def generate_samples(pipe, cache, directory, device, resolution, inference_steps):
    directory.mkdir(parents=True, exist_ok=True)
    pipe.unet.eval()
    # Decode explicitly: Diffusers only auto-upcasts an FP16 VAE, while this
    # experiment deliberately keeps the original SDXL VAE in FP32 with BF16
    # denoising latents.
    pipe.vae.to(device=device, dtype=torch.float32)
    rows = []
    for index, prompt in enumerate(PROMPTS):
        positive, negative, pooled, negative_pooled = [t.to(device) for t in cache["conditionings"][index]]
        seed = 3000 + index
        start = time.perf_counter()
        latents = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                     pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=negative_pooled,
                     height=resolution, width=resolution, num_inference_steps=inference_steps,
                     guidance_scale=5.0, generator=torch.Generator(device=device).manual_seed(seed),
                     output_type="latent").images
        decoded = pipe.vae.decode(latents.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        path = directory / f"prompt_{index}.png"
        image.save(path)
        rows.append({"prompt": prompt, "seed": seed, "path": str(path), "sha256": sha256(path),
                     "seconds": time.perf_counter() - start})
    return rows


@torch.no_grad()
def score_images(samples, records):
    """CPU metrics avoid changing the GPU training memory/timing measurement."""
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    images = [Image.open(row["path"]).convert("RGB") for row in samples]
    clip_processor = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION)
    clip = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION).eval()
    inputs = clip_processor(text=[row["prompt"] for row in samples], images=images, return_tensors="pt", padding=True)
    outputs = clip(**inputs)
    clip_values = (outputs.image_embeds * outputs.text_embeds).sum(dim=-1).tolist()
    del clip, clip_processor, inputs, outputs
    processor = AutoImageProcessor.from_pretrained(DINO_ID, revision=DINO_REVISION)
    dino = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION).eval()
    references = [Image.open(row["prepared"]).convert("RGB") for row in records if row["split"] == "train"]
    tensors = processor(images=references + images, return_tensors="pt")
    embeddings = F.normalize(dino(**tensors).last_hidden_state[:, 0], dim=-1)
    prototype = F.normalize(embeddings[:len(references)].mean(dim=0), dim=0)
    similarities = (embeddings[len(references):] * prototype).sum(dim=-1).tolist()
    del dino, processor
    return {"clip_text_image_cosine_mean": float(np.mean(clip_values)), "clip_per_prompt": clip_values,
            "dino_train_subject_cosine_mean": float(np.mean(similarities)), "dino_per_prompt": similarities,
            "caveat": "CLIP alignment and DINO similarity are proxies, not human-rated quality or definitive identity."}


def save_adapter(model, directory):
    state = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    torch.save(state, directory / "adapter.pt")
    return {"path": str(directory / "adapter.pt"), "sha256": sha256(directory / "adapter.pt"),
            "trainable_parameters": sum(t.numel() for t in state.values())}


@torch.no_grad()
def adapter_diagnostics(model):
    down_errors, magnitude_errors = [], []
    for module in model.modules():
        if isinstance(module, AdapterLinear):
            if module.use_nora:
                down_errors.append(float((module.effective_down().norm(dim=0) - 1).abs().max()))
            if module.use_dora:
                # Use FP32 effective weights for this diagnostic; exporting
                # into the BF16 base dtype introduces expected quantization.
                adapted = module.lora_B @ module.effective_down()
                adapted.add_(module.weight)
                merged = adapted * (module.m / module._row_norm(adapted))
                magnitude_errors.append(float((merged.norm(dim=1, keepdim=True) - module.m.abs()).abs().max()))
    return {"nora_column_norm_max_abs_error": max(down_errors, default=None),
            "dora_magnitude_max_abs_error": max(magnitude_errors, default=None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/sdxl"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trial-steps", type=int, default=50)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-lrs-json", type=Path,
                        help="JSON containing selected_lrs {method: rate}, chosen using seed42 validation only")
    parser.add_argument("--conditioning-cache", type=Path,
                        help="Read an existing cache plus matching .json metadata and SHA256")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = {name: Path(__file__).parent / name for name in ("sdxl.py", "adapters.py")}
    source_paths["dora.py"] = Path(__file__).parent.parent / "dora.py"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    # Reject an incompatible resume before regenerating image crops, replacing
    # their manifest, or touching CUDA. Completed research artifacts stay intact.
    if (args.output / "provenance.json").exists():
        previous = json.loads((args.output / "provenance.json").read_text())
        expected = {"model": MODEL_ID, "model_revision": MODEL_REVISION, "dataset": DATASET_ID,
                    "dataset_revision": DATASET_REVISION, "config": config,
                    "source_sha256": {name: sha256(path) for name, path in source_paths.items()}}
        if any(previous.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Existing output is incompatible; use a fresh output directory")
    elif any(args.output.glob("*/result.json")):
        raise RuntimeError("Existing results have no matching provenance; use a fresh output directory")
    torch.set_num_threads(8)
    records = prepare_dataset(args.output, args.resolution)
    if args.prepare_only:
        print("CPU dataset preparation complete", flush=True)
        return

    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    import diffusers
    import transformers
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    selected_lrs = None
    if args.selected_lrs_json is not None:
        selection = json.loads(args.selected_lrs_json.read_text())
        if selection.get("selection_seed") != 42:
            raise ValueError("This comparison reuses learning rates selected on seed42 validation")
        selected_lrs = selection["selected_lrs"]
        if any(method not in selected_lrs or selected_lrs[method] not in (1e-5, 1e-4) for method in args.methods):
            raise ValueError("Selected learning rates must cover requested methods and match the prespecified grid")
    provenance = {"model": MODEL_ID, "model_revision": MODEL_REVISION, "weight_variant": "fp16 converted to bf16",
                  "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
                  "torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": diffusers.__version__,
                  "transformers": transformers.__version__, "python": platform.python_version(),
                  "gpu": torch.cuda.get_device_name(device), "device": args.device,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
                  "clip": [CLIP_ID, CLIP_REVISION], "dino": [DINO_ID, DINO_REVISION],
                  "config": config,
                  "controls": {"rank": args.rank, "effective_scaling": 1, "adapter_dropout": 0,
                               "targets": TARGETS, "seed": args.seed, "image_order_seed": 2027 + args.seed - 42,
                               "training_noise_seed": 2028 + args.seed - 42,
                               "train_indices": [0, 1, 2], "validation_index": 3,
                               "test_index": 4, "lr_candidates": [1e-5, 1e-4], "optimizer": "AdamW",
                               "weight_decay": 0.01, "grad_clip": 1.0, "lr_schedule": "constant",
                               "min_snr_weighting": False, "prior_preservation": False,
                               "latent_cache": "deterministic VAE posterior mode; fixed center crop; no augmentation",
                               "train_precision": "BF16 frozen base / FP32 adapters / BF16 autocast"},
                  "limitations": ["Only five photos of one corgi with a shared orange backdrop; 3/1/1 split.",
                                  "One training seed per run; aggregate independent run directories for seed variability.",
                                  "Four fixed noise seeds and five timesteps probe one held-out photograph.",
                                  "Equal short validation-only learning-rate search; no other hyperparameter tuning.",
                                  "CLIP/DINO are imperfect proxies; generated images accompany metrics.",
                                  "This is a subject adaptation pilot, not a replication of the NoRA paper."]}
    provenance["lr_selection"] = ({"source": str(args.selected_lrs_json),
                                   "sha256": sha256(args.selected_lrs_json), "selected_lrs": selected_lrs,
                                   "selection_seed": 42} if selected_lrs is not None else
                                  {"selection_seed": args.seed, "source": "validation-only LR trials in this run"})
    provenance["conditioning_cache"] = (str(args.conditioning_cache) if args.conditioning_cache is not None else None)
    provenance["generation_precision"] = "BF16 frozen UNet with FP32 adapter branch; FP32 VAE decode"
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        for key in ("model", "model_revision", "dataset", "dataset_revision", "source_sha256", "config", "lr_selection"):
            if previous.get(key) != provenance.get(key):
                raise RuntimeError(f"Existing output is incompatible at {key}; use a fresh output directory")
    elif any(args.output.glob("*/result.json")):
        raise RuntimeError("Existing results have no matching provenance; use a fresh output directory")
    write_json(args.output / "provenance.json", provenance)
    source_dir = args.output / "executed_sources"
    source_dir.mkdir(exist_ok=True)
    for name, path in source_paths.items():
        (source_dir / name).write_bytes(path.read_bytes())
    print("Loading pinned SDXL pipeline", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, variant="fp16",
                                                    torch_dtype=torch.bfloat16, use_safetensors=True,
                                                    add_watermarker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.requires_safety_checker = False
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, revision=MODEL_REVISION, subfolder="scheduler")
    if scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("This experiment expects the epsilon prediction objective")
    cache = cache_conditioning(pipe, records, args.output, device, args.resolution, args.conditioning_cache)
    base_unet = pipe.unet.to("cpu")
    pipe.unet = None
    gc.collect()
    torch.cuda.empty_cache()

    def fresh_unet(method=None):
        pipe.unet = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        result = copy.deepcopy(base_unet).to(device)
        if method is not None:
            result = inject_adapters(result, method, rank=args.rank, targets=TARGETS)
            if not args.no_checkpointing:
                result.enable_gradient_checkpointing()
        pipe.unet = result
        return result

    baseline_dir = args.output / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    if not (baseline_dir / "result.json").exists():
        unet = fresh_unet()
        baseline = {"method": "baseline", "trainable_parameters": 0,
                    "validation": evaluate_mse(unet, scheduler, cache, 3, device),
                    "test": evaluate_mse(unet, scheduler, cache, 4, device)}
        baseline["samples"] = generate_samples(pipe, cache, baseline_dir / "samples", device,
                                               args.resolution, args.inference_steps)
        baseline["image_metrics"] = score_images(baseline["samples"], records)
        write_json(baseline_dir / "result.json", baseline)
        print("BASELINE", json.dumps({"test_mse": baseline["test"]["mean_mse"], **baseline["image_metrics"]}), flush=True)
        del unet

    for method in args.methods:
        directory = args.output / method
        directory.mkdir(exist_ok=True)
        if (directory / "result.json").exists():
            print(f"Skipping complete {method}", flush=True)
            continue
        trials = []
        for lr in (() if selected_lrs is not None else (1e-5, 1e-4)):
            trial_dir = directory / f"trial_lr_{lr:g}"
            trial_dir.mkdir(exist_ok=True)
            if (trial_dir / "result.json").exists():
                trial = json.loads((trial_dir / "result.json").read_text())
            else:
                unet = fresh_unet(method)
                trial = {"lr": lr, "training": train(unet, scheduler, cache, device, args.trial_steps,
                                                       lr, trial_dir / "training.jsonl", args.seed),
                         "validation": evaluate_mse(unet, scheduler, cache, 3, device)}
                write_json(trial_dir / "result.json", trial)
                del unet
            trials.append(trial)
        selected = (selected_lrs[method] if selected_lrs is not None else
                    min(trials, key=lambda row: row["validation"]["mean_mse"])["lr"])
        write_json(directory / "selection.json", {"rule": "reuse seed42 LR" if selected_lrs is not None else
                                                  "lowest validation noise MSE; test not consulted",
                                                   "trials": trials, "selected_lr": selected})
        unet = fresh_unet(method)
        before_hash = frozen_hash(unet)
        training = train(unet, scheduler, cache, device, args.steps, selected, directory / "training.jsonl", args.seed)
        after_hash = frozen_hash(unet)
        if before_hash != after_hash:
            raise RuntimeError("A frozen base parameter changed")
        result = {"method": method, "seed": args.seed, "selected_lr": selected, "trials": trials, "training": training,
                  "validation": evaluate_mse(unet, scheduler, cache, 3, device),
                  "test": evaluate_mse(unet, scheduler, cache, 4, device),
                  "checkpoint": save_adapter(unet, directory), "diagnostics": adapter_diagnostics(unet),
                  "frozen_sha256_before": before_hash, "frozen_sha256_after": after_hash}
        result["samples"] = generate_samples(pipe, cache, directory / "samples", device,
                                              args.resolution, args.inference_steps)
        result["image_metrics"] = score_images(result["samples"], records)
        write_json(directory / "result.json", result)
        print("RESULT", json.dumps({"method": method, "test_mse": result["test"]["mean_mse"],
                                     "seconds": training["seconds"], **result["image_metrics"]}), flush=True)
        del unet

    results = [json.loads((args.output / method / "result.json").read_text())
               for method in ("baseline", *args.methods)]
    write_json(args.output / "results.json", {"provenance": provenance, "results": results})
    if selected_lrs is None:
        selections = {method: args.output / method / "selection.json" for method in args.methods}
        write_json(args.output / "selected_lrs.json", {
            "selection_seed": args.seed,
            "rule": f"Reuse seed{args.seed} validation-only choices for additional training seeds",
            "selected_lrs": {method: json.loads(path.read_text())["selected_lr"]
                             for method, path in selections.items()},
            "selection_file_sha256": {method: sha256(path) for method, path in selections.items()},
        })
    canvas = Image.new("RGB", (320 * len(PROMPTS), 345 * len(results)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, result in enumerate(results):
        for col, sample in enumerate(result["samples"]):
            image = Image.open(sample["path"]).resize((320, 320))
            canvas.paste(image, (col * 320, row * 345 + 25))
            draw.text((col * 320 + 4, row * 345 + 5), f"{result['method']}: prompt {col}", fill="black")
    canvas.save(args.output / "sample_grid.jpg")
    print(f"Complete: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
