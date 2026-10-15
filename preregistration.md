# Preregistration

The complete design and 41-cell matrix are specified in the user-provided study protocol dated 2026-10-14. No primary method is removed for short-run weakness. HPO is 12 configurations per leaf (LR 1e-6/1e-5/1e-4/1e-3 × scale 0.25/1/4), 128 updates for screening, two full tuning configurations on seeds 1002/1003, then fresh seeds 2001–2003. Selection uses endpoint dev_tune loss only. `dev_audit`, GSM8K, and MATH-500 remain sealed until the registry and analysis freeze. All failures and pending jobs are retained in the coverage matrix.

Primary contrasts are fixed L vs fixed I, allocation within L and I, score families, restart tax, compression distortion/recovery, export quality, and equal-compute quality. Reports separate equal exposure, parameter budget, GPU-time, and export representation.

## Pipeline engineering decisions, 2026-10-15

Pipeline boundaries use the first intact effective update crossing the requested
token fraction, with the exact overshoot recorded. Pilot and student use separate
deterministic RNG streams and disjoint contiguous slices of the common manifest.
The pilot has a complete normalized allocation/LR schedule. The native student
schedule spans its full 80% portion even when common export cuts it at 75% of the
total pipeline. Wide sources use the full native LR horizon until their 75% cutoff;
ordinary recovery resets optimizer and LR schedule for the remaining updates.

Spectral global allocation uses a zero floor and singular-value prefix constraints;
custom online slot pruning retains its registered one-slot floor. Equal-cost
spectral allocation is exact, small heterogeneous cases use exact prefix DP, and
larger cases are explicitly labeled as a prefix value-per-cost heuristic.

The initial common-export engineering runner uses uniform retained ranks but can
underspend when a source module has fewer available directions. The corresponding
smoke runs remain excluded from primary selection and equal-size claims. The final
common-export allocation policy must be resolved and frozen before the primary
queue; eight successful pipeline executions do not establish this fairness gate.
