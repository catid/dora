# Implemented method matrix

The declarative registry in `configs/methods.yaml` contains all 41 required IDs. The current executable runner covers the 29 native cells (fixed, wide, reference AdaLoRA, online scores, allocation controls, triplets, triplet orthogonality, and uncertainty ablations). Restart and train/compress/recover IDs are registered but their multi-stage runners are not yet promoted; they remain pending/blocked in coverage.

The native runner uses BF16 frozen Qwen weights, FP32 adapter/master parameters, AdamW with zero weight decay, native SDPA, fixed q/o targets, deterministic seed streams, effective updates accumulated to at least 8192 input tokens, supervised-token loss normalization, and endpoint `dev_tune` NLL selection.
