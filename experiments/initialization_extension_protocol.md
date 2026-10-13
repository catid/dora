# Initialization extension protocol, 2026-10-09

This is a separate, source-reviewed extension of the accepted DoRA initialization study. It does not change the frozen `main_v2` registries, their selection, the accepted runtime, or public `dora.py`. Source and formula evidence is in `initialization_extension_review.md`. `init_extension.py:extension_families()` declares every setting; `init_extension_registry.py` crosses these settings with the complete task LR grids and binds this protocol before extension validation starts.

## Finite candidate opportunity

The six registered families are the one-setting `accepted_anchor`, four-setting `accepted_gaussian_scale` with {.25,.5,1,2} times the original draws, plus four settings each for `lora_null_task_tail`, `gme_paper`, `gme_source_v1`, and `gme_source_v2`. Null balance is {.25,.5,1,2}; GME tau_B is {3e-5,1e-4,3e-4,1e-3}. All other exact parameters appear in the registry, including paper/source moment policy and safeguards. GME has twelve settings across its three children; parent and child counts must be reported. Each setting crosses the complete task LR grid.

| Training task | Rank | LR grid | Fixed LR | Pilots |
|---|---:|---|---:|---:|
| SciFact | 8 | .0001, .001 | .001 | 42 |
| NFCorpus | 8 | .00003, .0001, .0003, .001 | .0001 | 84 |
| Aircraft | 2 | .00001, .0001, .0003, .001 | .001 | 84 |
| ViGGO extraction | 8 | .0001, .0003 | .0003 | 42 |

Total validation pilots: 252. NFCorpus-trained SciFact transfer is the fifth evaluation dataset, without extra fitting or selection. Pilot seed is 42; final seeds are 42, 43, 44. These results remain separate from both the main grid and fresh-seed confirmation. Candidates, formula branches and hyperparameters are not changed after validation or test inspection.

Use the same pinned models/data and audited splits, target modules, optimizer groups and order, losses, dropout, fitting budgets and checkpoint policies as the accepted task. Retrieval pilots use the full five-epoch budget and selected seed42 fits may be reused after selection. Aircraft uses 10-epoch pilots and fresh 20-epoch finals. ViGGO uses 48-update pilots and fresh 256-update finals. All extension pilots for a training task finish before selection and test evaluation. Fixed-rate selection tunes initialization; joint selection tunes initialization and LR. Selection metrics/tie breaks are unchanged: retrieval nDCG@10, Aircraft macro accuracy then CE, ViGGO NLL minimized, then lower LR and candidate ID.

## Accepted runtime and initial conditions

Retain W[out,in], up B[out,r], down A[r,in], column norm axis0, full denominator derivative, ambient norm arithmetic, dense projection and implicit scale1. MiniLM/Qwen use frozen BF16 W and FP32 factors/magnitude. Aircraft uses frozen FP32 W, FP32 factors/magnitude, and BF16 autocast. No denominator clamp, detached gradient, base recentering, rank redistribution, frozen factor, optimizer change or new trainable parameter is introduced. Every path consumes the accepted global CPU initialization draws before replacing factors with locally generated values, preserving task RNG trajectory.

LoRA-Null task-tail uses bottom-r eigenvectors U0 of the existing `corda_input_cov`, then the square-root SVD factors of W U0 U0.T, with reciprocal balance. The CPU FP32 thin projected SVD is algebraically equivalent and avoids the dense projected SVD. Both factors train. Runtime centers the initial product using the accepted `W+(BA-B0A0)` representation and original magnitude. Exact initial-forward parity is required. Missing source-normalized covariance is an error; there is no silent pooled-covariance fallback. The task-only calibration is a named adaptation. No preservation conclusion is claimed without separate knowledge data and retention evaluation.

GME uses a coordinated uncompensated warm start W+BA, retaining original-W magnitude throughout baseline, probes and final initialization. It intentionally changes initial outputs. Each recipe records before/after moments, initial logit/output drift, initial validation quality, response coefficients, solves/fallbacks, final factors and B/W norms. Zero BF16 probe response and singular response are retained as observations; they do not establish equalization. Nonfinite factors, model outputs or training states fail the candidate. The frozen source safeguards run as declared; no repair/retry with changed epsilon, tau, damping, seed or precision is permitted.

## Calibration and source distinctions

Use the existing fixed 128 training examples, calibration seed20261009 and batch size8. No validation/test activations or labels enter initialization. Reuse source-normalized per-forward input Grams for Null. These examples are not a representative external knowledge corpus. Record exact IDs, order, tokenizer, masks, tensor/calibration hashes and frozen model state. Aircraft rebinds initialization to each final seed's accepted random classifier as in the main study.

GME collects unlabeled eval-mode forward statistics on those same training inputs. Hook complete transformer-block outputs, with one moment domain per current checkpoint. MiniLM uses each encoder block and valid query/document token masks in the existing fixed calibration order. ViT includes CLS plus all patch tokens for each block. Qwen uses decoder block outputs and attention masks, including both prompt and target tokens exactly as present in existing training-calibration inputs; it does not substitute a final-normalized hidden state for a pre-final-norm block output. Every pass uses the same inputs/order/masks/autocast and records valid scalar counts.

Pool token-by-hidden scalar moments with FP64 raw sums, a recorded numerical adaptation to source FP32 accumulation. Paper skew uses mu3/(variance+eps)^1.5; V1/V2 use mu3/max((variance+eps)^1.5,eps). Targets use the FP32 lower median across the observed blocks. In all recipes use a private CPU factor generator seeded by fit seed, deterministic named-module order, FP32 returned factors, and one global draw stream through all blocks/probes. Preserve and restore model/RNG state and mode on callback errors.

Paper uses two probes and subtracts scale-only response from scale-plus-skew response for the skew coefficients. V2 instead subtracts baseline; V1 uses one diagonal probe. Paper coefficients are clipped ±10 and its Eq20 ridge pseudoinverse uses lambda1e-6 when determinant is below1e-6 or condition exceeds1e6, followed by a recorded diagonal fallback for invalid damped solutions. V2 uses determinant-only fallback and diagonal coefficient floors. Paper's Gamma draw is population-centered and uses gamma_min even at zero; source arms use empirical sample mean/std matching and best-skew retries, with Gaussian near-zero skew. Full options are frozen in each candidate; the recipes are never conflated.

The original GME text protocol used512examples/batch32, source rank/alpha, and PEFT DoRA geometry. Our fixed128/batch8, task mappings, FP64 sums, rank8/2, scale1 and accepted column DoRA are explicit adaptations. Gap-Init is excluded because none of the current checkpoints supplies aligned image/text representations; it is not simulated by unrelated gaps.

## Gates, immutable evidence and reporting

Before production fitting: independent CPU formula tests, registry Cartesian/count checks, source bindings, actual-model initial probe, accepted anchor RNG/output/trajectory parity, unchanged trainable parameter order/count, deterministic recreation and trained-checkpoint replay. Null requires centered parity; GME requires finite measured drift and unchanged base/magnitude. Source snapshots include backend, wrappers, protocol, registry builder, dependencies and accepted task source. No production manifest points to moving source files.

Retain every registered numerical failure with candidate/seed, stage, optimizer-step count, observation/log receipt and hashes. Infrastructure/programming errors stop work for correction and refreezing; they are not scored as negative algorithms. All-pilot failure yields no family selection; any failed final seed yields unavailable three-seed mean and paired interval. Save raw retrieval rankings, Aircraft class labels, and ViGGO outputs and rescore them independently.

Report all six families, parent/child pilot budgets, fixed/tuned selections, all three final seeds, means/sample SD and the established10000 crossed paired seed/example bootstrap intervals (Aircraft class-stratified), without multiplicity correction. No test-score filter picks displayed families. Compare to the same-LR or correspondingly selected extension accepted anchor. Equal-budget Gaussian comparisons may reuse audited provenance-matching main fits; missing exact matches remain unavailable. Do not pool observations across main/confirmation/extension. Clearly separate cold calibration/probe/decomposition time, cache hits and fit/evaluation time; retrieval receipts that start after injection are not full end-to-end timing.
