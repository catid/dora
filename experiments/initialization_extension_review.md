# Separate initialization extension: source review and implementation boundary

This review closes three leads that were unverified when the main initialization grid was frozen. It does not add candidates to that grid, revise its validation selection, or claim that these methods have been trained locally. `init_extension.py` is a separate initialization backend. Its CPU formula tests are engineering evidence only. Any downstream extension needs its own frozen registry and calibration/fit receipts before validation begins.

**Correction to the earlier discovery note:** GME-Init means **Gamma-Moment Equalization**. The description “geometry of audio task activations” in `initialization_literature.md` was incorrect. GME uses layer-output moments and was evaluated on text and vision-language models. The original note and frozen artifacts remain distinguishable from this later primary-source review.

## Primary evidence and version pins

Downloaded papers, extracted text, source trees, per-file hashes and HTTP failure receipts are retained under `/var/tmp/dora-bench/init_study/literature_extension`. `initial_manifest.json`, `additional_manifest.json`, `repository_manifest.json`, and `provenance_manifest.json` record provenance; `artifact_sha256.json` binds the final cache contents. Paper formulas were read from the full primary sources, rather than search summaries.

| Method | Primary paper | Inspected implementation |
|---|---|---|
| Gap-Init | [When Is Rank-1 Enough? Geometry-Guided Initialization for Parameter-Efficient Fine-Tuning](https://arxiv.org/html/2602.01522v1), §4 | [HaoranZhao2000/Gap-Init](https://github.com/HaoranZhao2000/Gap-Init/tree/f47debb15c56379d15137d019b9d608ec447b5d4), linked by the [author project page](https://haoranzhao.com/gap-init/); PEFT 0.17.1 is pinned in its environment. |
| GME-Init | [Gamma-Moment Equalization for LoRA Initialization in Parameter-Efficient Fine-Tuning](https://mdpi-res.com/d_attachment/entropy/entropy-28-00873/article_deploy/entropy-28-00873.pdf), Entropy 28(8), 873, DOI [10.3390/e28080873](https://doi.org/10.3390/e28080873), §3 | Author-associated [V2](https://github.com/HuiYulinn/GME-Init-V2/tree/3e0c746925c8745037f0f137d7e93fcbcee9d023) and [V1](https://github.com/HuiYulinn/GME-Init-V1/tree/b892f363a3b8f7d57f55291e00ec6f79f959bbbc). The account identifies YuHui Lin at JiMei University. Neither the inspected publication text nor repository metadata established a publication-to-repository link; do not call these a verified publication release. |
| LoRA-Null | [Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](https://arxiv.org/html/2503.02659v2), AAAI 2026 | [HungerPWAY/LoRA-Null](https://github.com/HungerPWAY/LoRA-Null/tree/1e6808abb81fe10e50b8172c40ac9a8ab4f11e83), linked from the paper's arXiv abstract. |

## Gap-Init: an explicit task boundary

In canonical notation, `W[out,in]`, up `B[out,r]`, down `A[r,in]`. For image/text representations in the same coordinates, the paper uses `g_l = mean_i(z_text_i,l - z_vision_i,l)`, sets the first up column to its normalized direction, and initializes down to zero. It discusses optional SPOT neighborhood smoothing. At higher rank the paper says remaining up columns are random; geometric layer selection is a separate structural intervention.

The inspected implementation has narrower behavior:

- `run_diagnostics_master.py` runs text through the language model and obtains an image anchor from Q-Former output, the existing language projector, and token averaging. Each text block's pooled output is compared with that same projected image anchor. Text pooling does not exclude padding. It averages batch-level gap vectors. SPOT is a separate scalar diagnostic, not smoothing applied to saved gap vectors.
- `train_caption_master.py:166–239` and the VQA counterpart write only `B[:,0] = g/(||g||+1e-6)` and zero A. Remaining B columns retain PEFT's zero initialization, unlike the paper's rank-greater-than-one random columns. Dimension mismatches are silently skipped in that source.
- The supplied `gap_init_rank1_naive.yaml` uses BLIP-2 OPT2.7B, rank 1, alpha 2, dropout .05, the last six blocks, and frozen Q-Former. Its extra alignment stage is disabled. The optional caption alignment-stage body is a `pass` placeholder, so it is not evidence of an executed auxiliary-loss recipe.

MiniLM retrieval, standalone ViT Aircraft, and text-only Qwen2.5-3B do not supply the required aligned image/text representations. Query/document differences, class differences, or a synthetic image anchor would be new derivatives. Qwen's grouped-query v-projection also has a different output width from its block hidden states; copying a block gap into that projection would be dimensionally invalid. Gap-Init is therefore **not applicable to the existing tasks without changing the task/model or naming a new method**. No fake Gap-Init arm is included in the extension.

## LoRA-Null: verified recipe and cheap projected factorization

Let `C = X_pre X_pre.T` denote an input activation moment. Take the bottom-r eigenvectors `U0` of C, the approximate activation null space. Define `T = W U0 U0.T`, then initialize from the rank-r SVD `T=U S Vh` using `B=U sqrt(S)`, `A=sqrt(S) Vh`. The source stores residual `W-T`; accepted local DoRA instead centers `BA` using its unchanged `W+(BA-B0A0)` runtime. This compensation representation and full denominator derivative remain local DoRA adaptations.

The implementation computes a thin SVD of `Z=W U0`: if `Z=Uz Sz Vzh`, then `B=Uz sqrt(Sz)` and `A=sqrt(Sz) Vzh U0.T`. Because U0 has orthonormal columns, these are the same projected singular components algebraically. This avoids a dense SVD of an output-by-input matrix; the input covariance eigensolve remains necessary. The independent test compares the product and factor Gram matrices with a separately constructed dense NumPy projector and SVD.

The official `adapterlib/decomposition.py:435–455` follows this projection recipe but forms T explicitly. Its names `ALinear`/`BLinear` are opposite the conventional up/down notation. The default `sigma_fuse='UV'` applies the square-root split. Root `step2.sh` launches both train-both v1 and freeze-one-factor v2; the extension implements **train-both v1 only**. Its optional reciprocal factor balance is a local conditioning control.

The original preservation experiment needs a separate representative knowledge corpus and retention evaluation. Source `step1.sh` uses rank 128 and 256 NQ Open calibration samples. `build_adapter.py` forwards its default seed **233** to the loader, overriding the loader's standalone default of 3. The loader concatenates training questions and samples fixed windows of length **2048**; the paper describes a **1024** maximum, a protocol discrepancy to retain. Source calibration divides each sequence's inputs by `abs(max(input))`, forms an unmasked Gram, and accumulates with divisor 256. Training scripts use 100,000 task examples and one epoch; TriviaQA, NQ Open and WebQuestions measure retention.

Using the current 128-example task calibration is a **task-tail initializer derivative**, not a reproduction of that preservation experiment. The extension requires the explicit `calibration_role='task_only'` or `'external_knowledge'` label. Even the latter does not establish retention without evaluation. Source-normalized task Grams and ordinary pooled task moments must have distinct configuration names. An eigenvalue tie at the rank boundary can make the selected subspace basis-dependent; the backend records the boundary gap.

## GME: output statistics, coordinated passes, and three explicit recipes

For each transformer block, pool all valid token-by-hidden scalar output entries over the calibration set. Compute variance `v_l` and standardized third central moment `g_l = mu3/(v_l+eps)^1.5`. Padding is excluded. Layer-output moments are scalars, not per-channel input covariances. Targets are the medians of block variances and block skews within a moment domain. Encoder and decoder domains should be separated; the current three checkpoints each have one relevant domain.

Adapters in every target block are probed together. Probe A draws down A with standard deviation `sigma_p` and zero target skew; up B is tiny Gaussian with standard deviation `tau_B/sqrt(r)`. Probe B uses the same down standard deviation and target skew `gamma_p`. The resulting downstream moments include upstream propagation; these are coordinated response estimates, not isolated per-adapter causal derivatives.

For baseline `(v0,g0)`, scale-only `(vA,gA)` and scale-plus-skew `(vB,gB)`, paper equations 9–14 are:

```text
a=(vA-v0)/(sigma_p^2+eps)  c=(gA-g0)/(sigma_p^2+eps)
b=(vB-vA)/(gamma_p+eps)    d=(gB-gA)/(gamma_p+eps)
[[a,b],[c,d]] [sigma^2,gamma].T = [v_target-v0,g_target-g0].T
```

Clip sigma and gamma. Draw centered Gamma down entries with shape `k=(2/max(abs(gamma),gamma_min))^2`, scale `theta=sigma/sqrt(k)`, and sign-corrected `Gamma(k,theta)-k*theta`. Paper Eq26 uses positive sign even when gamma is zero, retaining the gamma_min floor; only the source arms use a Gaussian when absolute gamma is below `1e-4`. Final up entries are Gaussian with standard deviation `tau_B*(sigma/sigma_p)/sqrt(r)`. Optional moment refinement multiplies sigma by `exp(eta_v*(v_target-v)/max(v_target,eps))`, adds `eta_g*(g_target-g)` to gamma, clips and resamples.

The three recipes deliberately stay separate:

| Recipe ID | Response formula | Sampling and safeguards |
|---|---|---|
| `paper` | Above two-probe marginal-skew equations, including cross terms. | One population-centered draw; no empirical sample matching. Paper Eq20 uses ridge pseudoinverse `(M.T M + lambda I)^-1 M.T`. Its §4 defaults are coefficient clipping ±10, determinant threshold `1e-6`, condition threshold `1e6`, and lambda `1e-6`, then diagonal fallback if the damped solution is invalid. These published safeguards are implemented and independently tested. |
| `source_v2` | Pinned V2 code uses **(probeB-baseline)/gamma_p** for b,d, instead of paper **(probeB-probeA)/gamma_p**. | Coefficients clipped to ±10; diagonal entries floored; determinant threshold `1e-6`; diagonal fallback. Each Gamma draw is empirically centered and rescaled to exact sample std; choose closest skew across four probe attempts or five final attempts. Source scale ratio is clipped to [.1,10]. |
| `source_v1` | One scale-plus-skew probe; independent diagonal variance and skew response estimates. | Same empirical sampler and scale-ratio clipping; no coupled solve. This is the older code recipe, not the publication's full two-probe algorithm. |

All three use fresh successive factor draws between probes, rather than silently reusing B noise. Source V2 parses only `encoder.layer.N` and reads `hidden_states[1:]`; its block grouping must be adapted for Qwen `model.layers.N` and ViT `blocks.N`. Our callback mapping makes the domain explicit. The source's `torch.median` is the **lower median** for an even number of blocks; this convention is retained and recorded for all three extension arms. The helper pools FP64 raw sums rather than the source's FP32 sums, an explicit numerical adaptation. For V1/V2, the callback must call `ScalarMoments.result(source_stability_floor=True)` to retain their denominator `max((var+eps)^1.5,eps)`; the paper arm uses `(var+eps)^1.5`.

Source V2's DoRA script uses 512 calibration examples, batch size 32, `tau_B=1e-4`, sigma [.005,.05], gamma ±.8, probe sigma .02, probe gamma .2, and two refinement steps with both rates .5. It targets RoBERTa blocks with task-specific rank/alpha. It constructs PEFT DoRA before replacing the factors, and does not reset its magnitude after probes or final factors. The extension retains **the original accepted DoRA magnitude** and local scale 1, with fixed rank and current targets. These are documented integration choices, not the original rank/alpha/task protocol.

GME requires a **nonzero warm start**. Centering `BA-B0A0` during its probes or final initialization would erase the output perturbations it tries to equalize. The source's B norm check only reports `||B||/||W|| < 1e-3`; it does not rescale B. The extension retains that diagnostic without treating it as proof of small functional change. BF16 addition can round small probe updates away, so zero response matrices and fallback counts must be reported, together with before/after moment and logit differences. No claim of achieved moment equalization follows from merely returning finite factors.

## Applicability and model callback contract

| Method | MiniLM retrieval | ViT Aircraft | Qwen2.5-3B ViGGO |
|---|---|---|---|
| Gap-Init | No aligned image/text model | No text representation | No vision representation |
| LoRA-Null task-tail | Input Gram available; named adaptation | Input Gram available; named adaptation | Input Gram available; named adaptation |
| LoRA-Null preservation protocol | Requires representative knowledge calibration and retention task | Requires a separate vision knowledge/retention design | Requires representative knowledge calibration and retention task |
| GME paper/V1/V2 | Transformer block outputs with valid-token mask; paired-query/document calibration must be fixed | ViT block outputs; all patch tokens and CLS token explicitly included | Decoder block outputs with attention mask; keep prompt/target token inclusion fixed before validation |

`initialize_gme(weights, ranks, module_layers, observe, ...)` returns CPU FP32 factor pairs and a JSON-serializable receipt. `observe(factors_or_None, stage)` must install all factors simultaneously, run fixed training-only calibration in eval mode, and return each declared block's `{var, skew, count}`. `None` requests the accepted zero-update DoRA forward, including its column normalization arithmetic; bypassing to raw Linear would change the BF16 baseline geometry. The backend rejects changed layer sets, scalar counts and nonfinite observations. It uses a private CPU generator, never consumes global RNG itself, and retains every probe/refinement result.

The runner must preserve and restore model buffers, factors, magnitude, mode and external RNG, including on callback errors. Do not alter the frozen accepted runtime. For a warm probe, set its centered reference contribution to zero or use the existing uncompensated warm-start route. Observe the declared transformer block outputs, not a mixture of individual target Linear outputs and final normalized outputs. MiniLM/Qwen keep BF16 base storage and FP32 factors/magnitude; Aircraft keeps FP32 base/factors/magnitude and BF16 autocast. The callback and fit must share the accepted task autocast policy.

Before validation, freeze model/data pins, calibration IDs/masks/order, domain/module mapping, statistic accumulation dtype, recipe options, factor seed, source hashes and the candidate list. Save factor products and initial logits/moments, compare baseline initial outputs with the accepted control, and verify trained-checkpoint replay. LoRA-Null centered gates require initial parity; GME records intentional initial differences instead of requiring parity. The source wrapper and this backend must be frozen separately from `main_v2`.

## Finite extension proposal, before any extension validation

`init_extension.extension_families()` declares the exact finite configuration dictionaries: `lora_null_task_tail` uses `corda_input_cov` (no silent fallback), task-only calibration, and reciprocal balance {.25,.5,1,2}; `gme_paper`, `gme_source_v1`, and `gme_source_v2` each use `tau_B` {3e-5,1e-4,3e-4,1e-3}, holding the other documented defaults fixed. Null's source-balance index is 2 and each GME source tau index is 1. GME parent budget is twelve initializer configurations; report parent/child opportunities explicitly. The accepted anchor and four-setting accepted Gaussian-scale family are additional controls with complete LR grids. Do not change candidates after viewing main or extension test scores.

The four training tasks retain rank 8 for MiniLM/Qwen and rank 2 for Aircraft. Full LR grids are SciFact [.0001,.001], NFCorpus [.00003,.0001,.0003,.001], Aircraft [.00001,.0001,.0003,.001], and ViGGO [.0001,.0003], with existing fixed rates .001, .0001, .001 and .0003 respectively. NFCorpus-trained SciFact transfer is the fifth evaluation dataset, not an additional fit. Including the one-setting accepted anchor and four Gaussian-scale controls, this is 21 initializer settings times the task LR grid: 42, 84, 84 and 42 validation pilots, 252 total. Calibration uses the existing 128 fixed training examples, a task-protocol adaptation; final seeds remain 42–44 and should be separate from main results.

This is a proposal until a distinct extension protocol/registry is frozen. It preserves the main protocol's validation-only selection, final seed handling, predictions, paired comparisons and failure retention; no main test scores choose extension formulas or bounds. Test results should not be pooled with main or fresh-seed confirmation results. Knowledge retention and multimodal Gap-Init remain separate task changes requiring their own protocols.

The backend's focused tests verify LoRA-Null against dense independent NumPy algebra; paper versus V1/V2 response equations; finite singular-response behavior; population Gamma moments and source empirical matching; masked pooling; coordinated callback order; and RNG isolation. They do not constitute model integration, validation or accuracy results. Cold calibration/decomposition/probe time and cached fitting time must be reported separately. GA/One/DA in the main study use an additional 128 labeled training-example gradient calibration with the original task loss; GME requires no labeled loss, and this difference must remain visible.
