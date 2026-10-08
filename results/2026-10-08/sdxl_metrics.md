| Method | Held-out denoising MSE ↓ | Original-prompt CLIP ↑ | Class-only CLIP, post hoc ↑ | DINO training-subject cosine ↑ |
|:--|--:|--:|--:|--:|
| Task baseline | 0.10471 | 0.3136 | 0.3163 | 0.0677 |
| LoRA | 0.10227 ± 0.00011 | 0.2846 ± 0.0077 | 0.2966 ± 0.0057 | 0.8025 ± 0.0117 |
| DoRA | 0.10226 ± 0.00012 | 0.2842 ± 0.0085 | 0.2955 ± 0.0067 | 0.8103 ± 0.0027 |
| NoRA | 0.10513 ± 0.00135 | 0.2692 ± 0.0110 | 0.2836 ± 0.0120 | 0.8675 ± 0.0026 |
| DoRA+NoRA | 0.10535 ± 0.00033 | 0.2781 ± 0.0042 | 0.2915 ± 0.0057 | 0.8657 ± 0.0083 |

Adapted results are mean ± sample SD over three training seeds. Each seed uses the same held-out noise probes and four predeclared generation prompts/seeds. DINO compares generated images to the three training photos; the shared orange backdrop can influence similarity. CLIP and DINO are separate proxies, not human judgments of quality or definitive identity. The seed-42 contact sheet in sdxl_samples.jpg accompanies these numbers.

Learning rates were selected using short 50-step validation trials, followed by fixed 500-step refits. This limited search does not establish each method's best achievable result; a rate selected at 50 steps may be suboptimal at 500. Final validation losses are retained alongside trial losses in tasks/sdxl.json.

The class-only CLIP diagnostic was added after training: scoring prompts replace the unfamiliar identifier phrase `sks dog` with `a dog`. Generated images, original CLIP measurements, denoising metrics, selected rates, and checkpoints are unchanged. This diagnostic was not used for model selection and remains an embedding proxy.
