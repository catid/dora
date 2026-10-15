# AdaLoRA reference source

Repository: https://github.com/QingruZhang/AdaLoRA

Revision: `d10f5ebee16c478fa2f41a44a237b38e8c9b0338`

Paths: `loralib/loralib/{adalora,layers}.py`, `loralib/LICENSE.md`.

Access: 2026-10-14; vendored 2026-10-15. Source files are copied verbatim; MIT
license retained. Integration lives outside these files in `v3/reference.py`.
The `SVDLinear` oracle disables merge-on-eval so the original backbone stays
unchanged. The actual adapter uses the same low-rank math with FP32 branch and
BF16 backbone. Original RankAllocator and compute_orth_regu execute directly.

The original allocator is step-based and threshold-based, including ties. Its
post-optimizer element scores and final every-step masking are kept separate
from the token-based, deterministic monotonic custom allocator. Ranknum stays
fixed; reference nominal scale is mapped to alpha = s * initial_rank and the
actual multiplier alpha/(ranknum+1e-5) is recorded. It has a small scalar-budget
mismatch from trainable E, which is reported rather than hidden.
