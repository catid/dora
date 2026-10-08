| Method | Flowers102 top-1 % ↑ | SciFact nDCG@10 ×100 ↑ | ViGGO exact match % ↑ | SDXL denoising MSE ↓ |
|:--|--:|--:|--:|--:|
| Task baseline | 98.88 ± 0.12 | 64.63 | 1.56 | 0.1047 |
| LoRA | 99.50 ± 0.02 | 65.10 ± 0.31 | 64.58 ± 4.32 | 0.1023 ± 0.0001 |
| DoRA | 99.58 ± 0.02 | 65.38 ± 0.32 | 64.84 ± 3.77 | 0.1023 ± 0.0001 |
| NoRA | 99.52 ± 0.15 | 68.47 ± 0.62 | 75.13 ± 3.63 | 0.1051 ± 0.0013 |
| DoRA+NoRA | 99.45 ± 0.24 | 68.63 ± 0.48 | 69.01 ± 6.79 | 0.1053 ± 0.0003 |

Values are means ± sample standard deviations over three training seeds for each adapted task. The vision baseline also has three seeds; other baselines are fixed. Each SDXL seed's MSE is averaged over 20 fixed noise/timestep probes on one held-out photo. The error bar measures variation across training seeds, not across those probes.

Baselines: frozen ViT backbone with a trained classifier; frozen MiniLM; frozen Qwen2.5-3B-Instruct; base SDXL.

All adapter methods use rank 8 and validation-only learning-rate selection. SDXL denoising MSE measures a noise-prediction objective, **not image quality**. The SDXL study contains only three training photos and one validation/test photo each of one subject. Flowers102 is near the pretrained model's accuracy ceiling. These tasks do not establish a universal method ranking.

Runtime fields in summary.json retain each task's recorded training wall-time scope; vision and JSON timing include validation. DoRA methods have extra magnitude parameters. Raw predictions, training logs, split manifests, and provenance are in raw_artifacts.tar.gz.
