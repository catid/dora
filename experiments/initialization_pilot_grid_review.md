# Reviewed initializer pilot grid proposal

This proposes four initialization configurations per family, each crossed with the same two task learning rates. The root registry remains authoritative. Every row includes the source factor recipe's default setting; main DoRA compensation, calibration batch size/masking, dense solvers and fixed trainable rank remain explicitly documented adaptations. An exact source factor default is not an exact reproduction of every paper's full training system.

| Family ID | Fixed recipe | Four settings (default emphasized) |
|---|---|---|
| pissa | top singular factors, factor_power=.5 | balance_gain .25, .5, **1**, 2 |
| milora | bottom singular factors, factor_power=.5 | balance_gain .25, .5, **1**, 2 |
| olora | Q[:,:r], R[:r] | balance_gain .25, .5, **1**, 2 |
| eva | centered, fixed rho=1, whiten=False | gain .5, **1**, 2, 4 |
| corda_ipm | top-r, adaptive_damping=True, source input moment | damping .001, **.01**, .1, 1 |
| astra_tail | bottom-r output moment | balance_gain .25, .5, **1**, 2 |
| loraga_default_direction | direction=ArB2r, stable both-factor scaling | gamma 4, **16**, 64, 256 |
| loraone_centered | negative-gradient principal factors, compensated derivative | gamma 32, **128**, 512, 2048 |
| filet | smallest Fisher energies, floor1e-8, source mean-of-batch moments | gain .5, **1**, 2, 4 |
| lorada_centered | gamma1024, damping1e-6, full training N, compensated derivative | variance_beta 0, .003, **.03**, .3 |
| cntlora_cross_centered | raw batch Grams, mode=cross, explicit target-minus-base delta | balance_gain .25, .5, **1**, 2 |
| inilora_prefit | group-mean rounded std, lr5e-4, StepLR5000/.9 | prefit_steps 0, 1000, 4000, **20000** |
| tlora_init_only | per-forward centered input Gram, fixed rank, both factors train | gain .5, **1**, 2, 4 |
| mica_init_only | bottom left singular up factor, zero down, both factors train | gain .5, **1**, 2, 4 |

`balance_gain` is reciprocal (B*=g, A/=g), preserving initial BA. `gain` multiplies both factors, changing BA by g² when both are nonzero. In a one-zero initializer, it scales the active factor once. The IniLoRA zero-step setting is a compensated random baseline within the same normal/group-std recipe; the full 20k-step source budget is retained.

Additional knobs must have separate named family IDs with the same four-setting opportunity, not silently expand selected rows after validation:

- `eva_whiten`: whiten=True, gain .5/1/2/4. Source PEFT divides by sqrt(activation singular value); count normalization stays fixed.
- `corda_kpm_task`: bottom-r selection, damping .001/.01/.1/1; task-calibration adaptation of original knowledge protocol.
- `astra_top`: top output eigenvectors, balance .25/.5/1/2.
- `loraga_A2rBr`, `loraga_ArBr`, `loraga_random`: each gamma4/16/64/256. Random partition uses a local generator and disjoint indices.
- `cntlora_self_centered`, `cntlora_shift_centered`: each reciprocal balance .25/.5/1/2. The source delta-sign ambiguity remains separately labeled. Opposite signs may be diagnostic arms with a predeclared budget, not selected retrospectively.
- `pissa_power`, `milora_power`: factor_power0/.25/.5/.75 with fixed balance1; these are local conditioning controls, not new named literature schemes.
- `inilora_lr`: fixed20000steps, lr .000125/.00025/.0005/.001. `inilora_scale`: fixedsourceprefitbudget, initial_std multiplier .5/1/2/4. These are optional declared full-cost families; do not suggest they ran merely because code can express them.

Random distribution/orientation/kernel families are defined separately by the root registry. Global parent-family maxima must report the total number of attempted child-family candidates, so splitting names does not conceal a larger tuning budget. Report fixed-original-LR and validation-tuned results separately; use identical final seeds and no test selection.

## Coverage that remains outstanding

The main backend centered-compensates all nonzero products. A separate `init_warmstart.py` backend now implements intentional uncentered LoRA-One/LoRA-DA/CNTLoRA products, with original-W magnitude by default and optional `adapted_column` magnitude. These lanes have no initial-function-parity gate; downstream fits remain outstanding. Use the same four settings as their centered counterparts with family IDs `loraone_warm`, `lorada_warm`, `cntlora_cross_warm`, `cntlora_self_warm`, `cntlora_shift_warm`, each crossing the same two learning rates. The official LoRA-One DoRA branch resets row magnitude, so reusing that verbatim would also change accepted column DoRA geometry.

The current TLoRA and MiCA arms only compare their initial subspaces with both factors trainable and fixed rank. Full TLoRA freezes A and reallocates ranks/scales from sensitivity; full MiCA freezes B. Full EVA redistribution (rho2), LoRA-SB's middle-matrix parameterization and LoRA-Pro's optimizer are also structural or optimization arms, not completed by these initializer derivatives.

LoftQ has a separate dense NF4 backend, source/quantizer parity and engineering audits. Its downstream task fits remain outstanding. Original-magnitude controls are primary; quantized-magnitude arms are distinct. For multi-iteration LoftQ, Q(W) is a quantizer baseline, while `base_source=paired_loftq` provides an additional control sharing its final optimized base.

Gap-Init, GME-Init and extreme-subspace leads still require primary-source recipe review. Specialized modality/domain statistics have not been collected. There is no claim that every discovered literature method has been tried or that the full user ablation is complete.
