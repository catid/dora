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

The first common-export engineering runner used uniform retained ranks and
underspent when source modules had fewer available directions. These smoke records
remain preserved and excluded from equal-size claims. Before primary execution,
the common deployment branch is now frozen to **global retained singular-value
energy** for every family, using the same prefix-feasible allocator and unchanged
original backbone. This redistributes ordinary ranks across modules instead of
silently exporting fewer factors. Equal-cost runs must fill the target budget;
heterogeneous integer underspend must be reported separately. Update numerical
rank remains distinct from allocated ordinary-factor rank and stored bytes.

In this new common pipeline, native uniform/global train-compress cells each
compress once at their natural 75% event, using the same common global operator.
Identical source/recovery branches may be reused physically with full standalone
accounting. Their native uniform/global comparison retains its distinct policies.
The common operator was chosen for a specified reconstruction objective and budget
adherence before primary outcomes; it is not a task-loss oracle or a selected winner.

## Synthetic controls

Before any corrected synthetic comparison, the feasible native budgets are fixed
at 16, 32 and 64 two-factor slot equivalents across eight 16×16 modules (average
rank 2, 4 and 8). The earlier plan's budget eight cannot accommodate even one
trainable triplet per module, so it is replaced rather than silently relaxing the
floor. Teachers use fixed orthonormal singular vectors, singular values linearly
spaced from 1 to 0.1 at ranks `[0,0,1,1,2,4,8,16]`, and construction seed 20261014.
Covariance is identity or a fixed rotated anisotropic transform with singular
values 3 to 0.3. Each paired data seed supplies 128 training and 512 independent
development draws per matrix. Training repeats that fixed draw for 100 updates;
noise standard deviation is 0 or 0.03. Peak LR 0.01, scale 1, primary schedules and
zero adapter weight decay are shared. Positive custom penalty is 0.001; reference
penalty is its documented 0.1. These common-config synthetic results are diagnostic
and do not select LLM configurations or establish tuned superiority. The oracle
uses the known teacher and covariance to allocate retained singular energy; it
is explicitly an oracle, not a learned competitor.
