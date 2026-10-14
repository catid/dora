# Current limitations

The pinned MetaMathQA revision supplies 12,904 deduplicated train groups after the required development reservations, rather than the requested 20,000. This is recorded as a blocked scope and no duplicates are added.

The host has four RTX PRO 6000 Blackwell Max-Q GPUs, not the target eight B300s. Native screening is therefore serialized through four independent one-GPU workers. No DDP result is claimed.

The final native HPO and full-horizon confirmation are still running. Restart, compression/recovery, export-normalized, equal-GPU-second, 7B, heterogeneous-module, official GSM8K/MATH, and DDP stages are not complete. No quality winner is claimed until those artifacts exist or are explicitly marked blocked.

The reference AdaLoRA arm follows the original zero-middle-coefficient behavior and unnormalized orthogonality penalty, while the pinned PEFT source exposes a different reset path. The exact compatibility note is in `reference_audit.md`.
