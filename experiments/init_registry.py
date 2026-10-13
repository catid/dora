"""Build finite, immutable initialization grids before any main-study selection."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

from experiments import init_common as common


TASKS = {
    'scifact': {'rank': 8, 'fixed_lr': .001, 'learning_rates': [.0001, .001]},
    'nfcorpus': {'rank': 8, 'fixed_lr': .0001, 'learning_rates': [.00003, .0001, .0003, .001]},
    'aircraft': {'rank': 2, 'fixed_lr': .001, 'learning_rates': [.00001, .0001, .0003, .001]},
    'extraction': {'rank': 8, 'fixed_lr': .0003, 'learning_rates': [.0001, .0003]},
}
GAINS = [.25, .5, 1., 2.]


def families(lane='centered'):
    result = []

    def add(name, parent, kind, configs, note, default_index=None, **metadata):
        assert name not in {x['family_id'] for x in result}
        assert len(configs) in (1, 4)
        assert len({common.digest_json(x) for x in configs}) == len(configs)
        result.append(dict(family_id=name, parent_family=parent, kind=kind,
                           configs=configs, source_default_index=default_index,
                           interpretation=note, **metadata))

    add('accepted_anchor', 'accepted', 'random', [{'mode': 'accepted'}],
        'Exact accepted random-up/zero-down global-stream initialization.', 0)
    if lane == 'quantized':
        for iterations in (1, 3):
            for policy in ('original', 'quantized'):
                base = dict(bits=4, blocksize=64, iterations=iterations, magnitude_policy=policy,
                            adapter_scale=1.)
                control = f'nf4_control_i{iterations}_{policy}'
                add(control, 'nf4_quantizer_control', 'quantized',
                    [dict(base, method='quantizer_only', base_source='direct' if iterations == 1 else 'paired_loftq')],
                    'Same final dense NF4 base as paired LoftQ; accepted one-zero factors.', 0)
                add(f'loftq_i{iterations}_{policy}', 'loftq', 'quantized',
                    [dict(base, method='loftq', balance_gain=g) for g in GAINS],
                    'Dense dequantized LoftQ DoRA adaptation; source NF4 default is 1 iteration, block64.',
                    2 if iterations == 1 else None, comparison_family_id=control)
        return result
    if lane == 'warmstart':
        add('loraone_warm', 'loraone', 'warmstart',
            [dict(method='loraone', gamma=g, magnitude_policy='original') for g in (32, 128, 512, 2048)],
            'Intentional negative-gradient start; accepted column DoRA, original-W magnitude.', 1)
        add('lorada_warm', 'lorada', 'warmstart',
            [dict(method='lorada', gamma=1024, variance_beta=b, damping=1e-6, magnitude_policy='original')
             for b in (0., .003, .03, .3)],
            'Intentional DA update; full training N, eval-mode common calibration, column DoRA.', 2)
        for mode in ('cross', 'self', 'shift'):
            add(f'cntlora_{mode}_warm', 'cntlora', 'warmstart',
                [dict(method='cntlora', mode=mode, delta_sign='target_minus_base', balance_gain=g,
                      magnitude_policy='original') for g in GAINS],
                'Intentional domain-shift start with explicit target-minus-base sign; source sign unresolved.', 2)
        return result
    assert lane == 'centered'
    add('accepted_gaussian_scale', 'accepted', 'random',
        [dict(mode='accepted_scaled', scale=g) for g in GAINS],
        'Equal four-setting scale search around exact original CPU draws; includes scale1.', 2)
    noises = [
        ('normal', {}), ('uniform', {}), ('rademacher', {}), ('laplace', {}),
        ('student_t3', {'family': 'student_t', 'df': 3.}),
        ('student_t5', {'family': 'student_t', 'df': 5.}),
        ('student_t10', {'family': 'student_t', 'df': 10.}),
        ('truncnormal1', {'family': 'truncnormal', 'truncation': 1.}),
        ('truncnormal2', {'family': 'truncnormal', 'truncation': 2.}),
        ('sparse01', {'family': 'sparse', 'sparsity': .01}),
        ('sparse10', {'family': 'sparse', 'sparsity': .1}),
        ('sparse50', {'family': 'sparse', 'sparsity': .5}),
        ('orthogonal', {}), ('dct', {}), ('cropped_hadamard', {}),
    ]
    for orientation in ('up', 'down'):
        for name, extra in noises:
            family = extra.get('family', name)
            base = dict(mode='controlled', orientation=orientation, family=family, **{k:v for k,v in extra.items() if k!='family'})
            add(f'{name}_{orientation}', f'random_{family}', 'random',
                [dict(base, scale=g) for g in GAINS],
                'Local paired stream, analytic entry RMS scale/sqrt(rank), one zero factor.')
        add(f'spherical_{orientation}', 'spherical', 'random',
            [dict(mode='controlled', orientation=orientation, radial='unit_vectors', scale=g) for g in GAINS],
            'One-time rank-vector normalization only; no per-forward NoRA.')
        add(f'normal_sample_rms_{orientation}', 'random_normal', 'random',
            [dict(mode='controlled', orientation=orientation, normalization='sample_rms', scale=g) for g in GAINS],
            'Exact global sample RMS; separated from analytic expected RMS.')
        for permute in (False, True):
            add(f'bimi_{orientation}'+('_permuted' if permute else ''), 'bimi', 'random',
                [dict(mode='controlled', orientation=orientation, family='bimi', permute=permute, scale=g) for g in GAINS],
                'Repeated identity with optional explicitly recorded signed permutation.',
                2 if orientation == 'down' and not permute else None)
        for kernel in ('rbf', 'exponential', 'matern32'):
            add(f'normal_{kernel}_{orientation}', f'kernel_{kernel}', 'random',
                [dict(mode='controlled', orientation=orientation, kernel=kernel, kernel_length=k)
                 for k in (.25, .5, 1., 2.)],
                'Unit-diagonal covariance on arbitrary rank coordinates; gain1, lengths tuned; no neuron geometry.')
    add('lora_kaiming_uniform_down', 'lora_standard', 'random',
        [dict(mode='controlled', orientation='down', family='uniform', scaling='fan_in', scale=g/math.sqrt(3)) for g in GAINS],
        'PEFT-style uniform[-1/sqrt(in),1/sqrt(in)] at gain1, zero up; local RNG adaptation.', 2)
    add('normal_fanin_down', 'random_normal', 'random',
        [dict(mode='controlled', orientation='down', scaling='fan_in', scale=g) for g in GAINS],
        'Input-width scaling isolates the standard-down scale convention.')
    add('peft_complementary', 'peft_orthogonal', 'random',
        [dict(mode='controlled', family='peft_orthogonal', both_nonzero=True, scaling='absolute',
              scale=g, companion_scale=1/g) for g in GAINS],
        'Even-rank complementary subspaces, native /10 draws; centered for numerical startup identity.', 2)
    add('nonzero_uniform_original_in', 'nonzero_lora', 'random',
        [dict(mode='controlled', family='uniform', both_nonzero=True, scaling='original_input',
              scale=g/math.sqrt(3), companion_scale=g/math.sqrt(3)) for g in GAINS],
        'Nonzero LoRA AB_1_1_RESET: both factors uniform bounds gain/sqrt(original input); centered.', 2)
    add('inilora_alpha', 'inilora', 'random',
        [dict(mode='controlled', family='normal', both_nonzero=True, scaling='absolute',
              scale=g, companion_scale=g) for g in (.125, .25, .5, 1.)],
        'Both factors Gaussian std.5 at source setting; centered compensation.', 2)
    add('inilora_beta', 'inilora', 'random',
        [dict(mode='controlled', family='normal', both_nonzero=True, scaling='fan_in',
              scale=g*math.sqrt(2), companion_scale=g*math.sqrt(2)) for g in GAINS],
        'Both factors Kaiming normal fan-in, source gain sqrt2; centered compensation.', 2)
    for method in ('pissa', 'milora', 'olora'):
        add(method, method, 'data', [dict(method=method, balance_gain=g) for g in GAINS],
            'Source factor recipe; reciprocal factor balance tuning; centered representation adaptation.', 2)
    for method in ('pissa', 'milora'):
        add(f'{method}_power', method, 'data',
            [dict(method=method, factor_power=p) for p in (0., .25, .5, .75)],
            'Explicit local singular-value exponent split; source equal split .5.', 2)
    for whiten in (False, True):
        add('eva_whiten' if whiten else 'eva', 'eva', 'data',
            [dict(method='eva', centered=True, whiten=whiten, gain=g) for g in (.5, 1., 2., 4.)],
            'Fixed-rank activation PCA; no rho2 redistribution; source whitening formula when enabled.', 1 if not whiten else None)
    for selection in ('top', 'bottom'):
        add('corda_ipm' if selection == 'top' else 'corda_kpm_task', 'corda', 'data',
            [dict(method='corda', selection=selection, adaptive_damping=True,
                  covariance_key='corda_input_cov', damping=d) for d in (.001, .01, .1, 1.)],
            'Source per-forward normalized input Grams; task-data KPM is a knowledge-data adaptation.', 1)
        add('astra_top' if selection == 'top' else 'astra_tail', 'astra', 'data',
            [dict(method='astra', selection=selection, covariance_key='astra_output_cov', balance_gain=g) for g in GAINS],
            'Source normalized output batch Grams; source selection is tail.', 2 if selection == 'bottom' else None)
    for direction in ('ArB2r', 'A2rBr', 'ArBr', 'random'):
        add(f'loraga_{direction}', 'loraga', 'data',
            [dict(method='loraga', direction=direction, gamma=g) for g in (4., 16., 64., 256.)],
            'Author disjoint gradient-SVD directions, no singular-value factor; source default ArB2r gamma16.',
            1 if direction == 'ArB2r' else None)
    add('loraone_centered', 'loraone', 'data',
        [dict(method='loraone', gamma=g) for g in (32, 128, 512, 2048)],
        'Centered derivative cancels intentional starting gradient step; separate warmstart arm.', 1)
    add('filet', 'filet', 'data',
        [dict(method='filet', gain=g, input_covariance_key='filet_input_cov',
              output_covariance_key='filet_output_grad_cov') for g in (.5, 1., 2., 4.)],
        'Source normalized Gram columns and mean batch moments; centered.', 1)
    add('lorada_centered', 'lorada', 'data',
        [dict(method='lorada', gamma=1024, damping=1e-6, variance_beta=b) for b in (0., .003, .03, .3)],
        'Centered derivative; full training N; common eval-mode calibration is a source protocol adaptation.', 2)
    for mode in ('cross', 'self', 'shift'):
        add(f'cntlora_{mode}_centered', 'cntlora', 'data',
            [dict(method='cntlora', mode=mode, delta_sign='target_minus_base', balance_gain=g) for g in GAINS],
            'Centered derivative, raw per-batch Grams, fixed rank, explicit target-minus-base sign.', 2)
    add('inilora_prefit', 'inilora', 'data',
        [dict(method='inilora', prefit_steps=n, prefit_lr=.0005, prefit_device='cuda',
              initial_std_policy='group_mean_rounded') for n in (0, 1000, 4000, 20000)],
        'Source group std and Adam/StepLR; source budget20k retained, TF32 off; preprocessing timed.', 3)
    for method in ('tlora', 'mica'):
        add(f'{method}_init_only', method, 'data',
            [dict(method=method, gain=g, **({'covariance_key':'tlora_input_cov'} if method=='tlora' else {}))
             for g in (.5, 1., 2., 4.)],
            'Initializer subspace derivative; both factors train, fixed rank; source freezes/allocates differently.', 1)
    return result


def build(task, protocol, lane='centered'):
    settings = TASKS[task]
    definitions = families(lane)
    candidates = []
    lookup = {}
    for family in definitions:
        for index, config in enumerate(family['configs']):
            for lr_index, lr in enumerate(settings['learning_rates']):
                cid = f"{family['family_id']}__i{index}__lr{lr_index}"
                lookup[(family['family_id'], index, lr)] = cid
                candidates.append(dict(candidate_id=cid, family_id=family['family_id'],
                                       parent_family=family['parent_family'], initializer_index=index,
                                       source_default_setting=index == family['source_default_index'],
                                       spec={'kind':family['kind'], 'config':config}, learning_rate=lr))
    control = lookup[('accepted_anchor', 0, settings['fixed_lr'])]
    definition_map = {x['family_id']:x for x in definitions}
    for row in candidates:
        family = definition_map[row['family_id']]
        comparator = family.get('comparison_family_id', 'accepted_anchor')
        row['comparison_control_id'] = lookup[(comparator, 0, row['learning_rate'])]
    return dict(schema='dora-init-study-v1', study_lane=lane, task=task, **settings,
                seed=42, seeds=[42,43,44], protocol_sha256=common.sha(protocol), control_id=control,
                candidates=candidates, candidate_counts=dict(Counter(x['family_id'] for x in candidates)),
                parent_candidate_counts=dict(Counter(x['parent_family'] for x in candidates)),
                family_definitions=definitions, registry_builder_sha256=common.sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lane', choices=('centered','warmstart','quantized'), default='centered')
    args = parser.parse_args()
    for task in TASKS:
        registry = build(task, args.protocol, args.lane)
        path = args.output/args.lane/f'{task}.json'
        common.write_once(path, registry)
        common.load_registry(path, args.protocol, task)
        print(f'{args.lane} {task}: {len(registry["family_definitions"])} families, {len(registry["candidates"])} pilots; {path}')


if __name__ == '__main__':
    main()
