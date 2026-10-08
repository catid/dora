# Reproduce the task comparisons

The experiments use the shared adapters in `adapters.py`, checked against independent forward and gradient references. NoRA implements equations 7–9 of [Normalized Low-Rank Adaptation](https://arxiv.org/abs/2608.31036); DoRA + NoRA follows the paper's DoRA-style parameterization. These are controlled small-task comparisons, not reproductions of the paper's benchmark scores.

All four methods use rank 8, effective scaling 1, the same Kaiming-initialized input-to-rank factor, a zero output factor, no adapter dropout, and matched pretrained checkpoints. NoRA normalizes the input factor across rank during every forward pass, including its derivative. DoRA learns one output magnitude and detaches the adapted-weight norm. The experimental factor convention differs from the historical standalone demo so the initialization is consistent across methods. DoRA methods have additional magnitude parameters; exact counts are recorded.

## Environment

The measured environment is recorded in [environment.json](environment.json). Install a CUDA-enabled PyTorch build appropriate to the GPU. The pinned package set used here is in [requirements-experiments.txt](../requirements-experiments.txt):

```bash
python -m pip install -r requirements-experiments.txt
python -m unittest -v test_dora experiments.test_adapters
```

Use a filesystem with room for pretrained weights. On the measured host, artifacts and caches live under `/var/tmp/dora-bench`, separate from the nearly full `/home` filesystem. Every run records its model/dataset revisions, split manifests, source hashes, and configuration. Use fresh output directories when changing configuration.

## Tasks

| Task | Model and data | Final training budget | Selection |
| --- | --- | --- | --- |
| Vision | ViT-B/16 pretrained on ImageNet21k then ImageNet1k; official Flowers102 splits: 1,020 train / 1,020 validation / 6,149 test | 30 epochs, batch 64; attention and MLP adapters; classifier also trained | Equal 5-epoch trials at adapter LR 1e-4 / 1e-3; head LR 1e-3; best validation accuracy, then cross-entropy |
| Retrieval | MiniLM-L6-v2; SciFact: 726 train / 81 validation / 300 test queries; 5,183 documents | 5 epochs, batch 32; query/value adapters; supervised contrastive loss | Equal full-budget trials at LR 1e-4 / 1e-3; validation nDCG@10 |
| JSON extraction | Qwen2.5-3B-Instruct; public ViGGO utterances paired with their meaning representations, reversed into JSON extraction; 1,024 train / 128 validation / 256 test | 256 updates, batch 8; query/value adapters; target-token cross-entropy | Equal 48-update trials at LR 1e-4 / 3e-4; validation target-token NLL; final checkpoint selected by validation NLL |
| Subject adaptation | SDXL base 1.0; five public photos of one corgi, split 3 train / 1 validation / 1 test | 500 updates, batch 1, 1,024-pixel images; attention adapters | Equal 50-update trials at LR 1e-5 / 1e-4; validation denoising MSE |

Learning-rate selection uses seed 42, with selected rates reused for final seeds 42, 43, and 44. Test scores do not select learning rates or checkpoints. Retrieval reuses the winning seed-42 tuning checkpoint because its pilot and final budgets are identical. The vision baseline trains only its classifier; the other task baselines are frozen pretrained models.

The retrieval split excludes two official training queries duplicated in the test set. Extraction excludes duplicate utterances after lowercasing and normalizing whitespace, within and across official splits, before deterministic subsampling. Its training loss masks prompt tokens and supervises only the target JSON and end token. Test generation is greedy, capped at 256 new tokens. JSON exact match compares parsed objects, ignoring JSON formatting and key order but requiring exact keys and string values, including the dialogue act. The secondary slot F1 in the raw extraction results is gated on schema validity.

The SDXL denoising metric averages 20 fixed noise/timestep probes on the single held-out photo. It measures the denoising objective, not image quality. Generated images, CLIP text alignment, and DINO similarity are separate diagnostics. The five photos share an orange backdrop, so subject similarity can also reward learning the background. No class-image prior preservation or image augmentation is used.

## Commands

Run from the repository root. These commands select physical GPU IDs explicitly; run only on available GPUs.

```bash
export HF_HOME=/var/tmp/dora-bench/cache/huggingface
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python -m experiments.vision --seeds 42 43 44 --tune-lrs 0.0001 0.001
CUDA_VISIBLE_DEVICES=1 python -m experiments.retrieval
CUDA_VISIBLE_DEVICES=2 python -m experiments.extraction
CUDA_VISIBLE_DEVICES=1 python -m experiments.sdxl --seed 42
```

SDXL seeds 43 and 44 reuse seed 42's selected learning rates and conditioning cache. After the seed-42 run has produced `selected_lrs.json`:

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.sdxl --seed 43 \
    --output /var/tmp/dora-bench/sdxl_seed43 \
    --selected-lrs-json /var/tmp/dora-bench/sdxl/selected_lrs.json \
    --conditioning-cache /var/tmp/dora-bench/sdxl/conditioning_1024.pt
CUDA_VISIBLE_DEVICES=0 python -m experiments.sdxl --seed 44 \
    --output /var/tmp/dora-bench/sdxl_seed44 \
    --selected-lrs-json /var/tmp/dora-bench/sdxl/selected_lrs.json \
    --conditioning-cache /var/tmp/dora-bench/sdxl/conditioning_1024.pt
```

The exact commands and cache fingerprints are recorded with each run. Training checkpoints and full-resolution generations remain in the artifact directories; the committed report includes compact results, generated sample grids, source snapshots, and compressed raw metric logs.

After all three SDXL runs finish, compute the additional CPU-only CLIP diagnostic required by the report:

```bash
python -m experiments.sdxl_clip_diagnostic --run-dirs \
    /var/tmp/dora-bench/sdxl \
    /var/tmp/dora-bench/sdxl_seed43 \
    /var/tmp/dora-bench/sdxl_seed44
```

This post-hoc diagnostic replaces `sks dog` with `a dog` in CLIP's scoring text, uniformly for every saved image. The external CLIP encoder has not learned the subject identifier `sks`, so the original scores also reflect that identifier's pretrained meaning. `class_clip_metrics.json` retains the original scores alongside the class-only scores; generation and model selection are unchanged. Both versions are embedding proxies, not human image-quality ratings. Add `--wait` to the command to let the scorer wait for unfinished SDXL runs.

After all 12 extraction runs complete, the CPU-only paired analysis independently checks generated JSON and resamples matched training seeds and shared test examples:

```bash
python -m experiments.analyze_extraction
```

It writes `paired_bootstrap.json` and compressed sample arrays under the extraction artifact directory. Its 95% percentile intervals use 10,000 resamples and are exploratory: three training seeds and one held-out test set do not establish broad statistical or task-independent superiority. These intervals are separate from the report's standard-deviation error bars.

```bash
python -m experiments.report
```

The report generator refuses incomplete comparisons. It produces a table, an editable SVG and PNG chart, per-task JSON, and a compressed raw-artifact archive under `results/2026-10-08`. Error bars show standard deviation across training seeds, not uncertainty over test examples. The tasks use different quality metrics and are not combined into an overall score.
