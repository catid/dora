# Current limitations

The corrected v2 seal contains the requested 20,000 training rows, but they span 8,899 connected question groups because the source includes multiple augmentations per underlying problem. Development groups remain disjoint. Ten eligible groups were excluded by the held-out near-overlap filter.

The host has four RTX PRO 6000 Blackwell Max-Q GPUs, not the target eight B300s. Native screening is therefore serialized through four independent one-GPU workers. No DDP result is claimed.

Corrected v2 native HPO and all downstream confirmation stages are incomplete. The first corrected fixed-cell HPO values are screening measurements, not quality conclusions. Restart, compression/recovery, export-normalized, equal-GPU-second, 7B, heterogeneous-module, official GSM8K/MATH, and DDP stages are not complete. No quality winner is claimed until those artifacts exist or are explicitly marked blocked.

The reference AdaLoRA arm follows the original zero-middle-coefficient behavior and unnormalized orthogonality penalty, while the pinned PEFT source exposes a different reset path. The exact compatibility note is in `reference_audit.md`.
