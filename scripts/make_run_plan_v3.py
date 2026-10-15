#!/usr/bin/env python3
"""Complete bounded protocol plan, with explicit unresolved selection roles.

Conditional jobs are declared now and resolved from dev_tune only. They are not
quietly removed when one family has weak short-horizon performance.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path

import yaml


ROLES = {
    'FIXED_L': ['L_FIXED'], 'FIXED_I': ['I_FIXED'], 'REFERENCE': ['ADALORA_REF'],
    'ONLINE_L': ['L_RAW','L_GATE','L_DISP','L_ACT','L_RAW_UNCERTAINTY','L_GATE_UNCERTAINTY'],
    'ONLINE_I': ['I_RAW','I_GATE','I_DISP','I_ACT','I_RAW_UNCERTAINTY','I_GATE_UNCERTAINTY'],
    'TRIPLET_T0': ['T0_GATE','T0_GATE_ORTHO'], 'TRIPLET_T1': ['T1_GATE','T1_GATE_ORTHO'],
    'RESTART_L': ['L_PILOT_L_STUDENT','I_PILOT_L_STUDENT'],
    'RESTART_I': ['L_PILOT_I_STUDENT','I_PILOT_I_STUDENT'],
    'COMPRESS_L': ['L_UNIFORM_SVD','L_GLOBAL_SVD'],
    'COMPRESS_I': ['I_UNIFORM_SVD','I_GLOBAL_SVD'],
}


def trial_design(method):
    trials=[]
    for i,lr in enumerate([1e-6,1e-5,1e-4,1e-3]):
        for j,scale in enumerate([.25,1.,4.]):
            trial_lr=lr
            penalty=([1e-4,1e-3,1e-2][(i+j)%3] if method.endswith('_ORTHO')
                     else [.01,.1,1.][(i+j)%3] if method=='ADALORA_REF' else 0.)
            canonical=False
            # Documented original MNLI optimizer LR/alpha/penalty. Rank and
            # token horizon stay the study's: this is not a paper reproduction.
            if method=='ADALORA_REF' and i==3 and j==1:
                trial_lr,penalty,canonical=8e-4,.1,True
            trials.append({'trial':len(trials),'lr':trial_lr,'scale':scale,'penalty':penalty,
                           'canonical_reference_replacement':canonical})
    return trials


def build_plan(methods):
    rows=[]
    def add(rid,stage,method=None,role=None,seed=None,updates=0,deps=(),**extra):
        if method is not None and method not in methods: raise ValueError(method)
        rows.append({'run_id':rid,'stage':stage,'method_id':method,'selection_role':role,
                     'seed':seed,'updates':updates,'depends_on':list(deps),
                     'exposure_manifest':f'artifacts/manifests_v3/{stage}/seed{seed}.json' if updates else None,
                     'token_cap':{'mode':'exact_sealed_manifest','updates':updates,'nominal_input_tokens_per_update':8192,'maximum_overshoot_per_update':2047} if updates else None,
                     'expected_output':f'runs/{rid}/metadata.json','status':'pending',**extra})
    for gate in ['algebra','upstream_parity','allocator','data_labels','checkpoint_resume','compaction',
                 'gpu_all_representations','ddp_two_rank','sdpa_oracle_profile','pipeline_restart',
                 'pipeline_compression','answer_parser','diagnostic_restoration']:
        add(f'A_gate_{gate}','A_correctness',kind='test_gate')
    for method in methods:
        for seed in [901,902]:
            for cov in ['isotropic','anisotropic']:
                for noise in [False,True]:
                    add(f'A_syn_{method}_{seed}_{cov}_{noise}','A_synthetic',method,seed=seed,
                        budgets=[8,16,32],covariance=cov,noise=noise,teacher_ranks=[0,0,1,1,2,4,8,16],
                        oracle_control=True,steps=100,device='cpu')
    for method in ['L_FIXED','I_GATE']:
        add(f'A_full_horizon_{method}','A_full_horizon_diagnostic',method,seed=1001,updates=512,
            lr=1e-4,scale=1.,selection_eligible=False,partial_dev_groups=16,
            purpose='full-horizon allocation/recovery and throughput validation')
    gates=[r['run_id'] for r in rows if r['stage']=='A_correctness']
    def tournament(prefix,entries,horizon,search_seed,tune_seeds,report_seeds,rank=8,model='1.5B',targets='q_o'):
        terminals=[]
        for entry in entries:
            method=entry if entry in methods else None
            role=None if method else entry
            hpo=[]
            for trial in range(12):
                rid=f'{prefix}_hpo_{entry}_c{trial:02d}'
                add(rid,prefix+'_hpo',method,role,search_seed,horizon//4,gates+(['selection_freeze'] if role else []),trial_index=trial,
                    trial_config=trial_design(method)[trial] if method else {'resolve_from_selected_method_design':trial},
                    rank=rank,max_rank=2*rank,model=model,targets=targets)
                hpo.append(rid)
            rescue=[]
            for offset in [1,2]:
                rid=f'{prefix}_rescue_{entry}_{offset}'
                add(rid,prefix+'_rescue',method,role,search_seed,horizon//4,hpo,
                    conditional='best_valid_lr_is_boundary_or_all_failed',outward_power_of_ten=offset,
                    rank=rank,max_rank=2*rank,model=model,targets=targets)
                rescue.append(rid)
            tuning=[]
            for config_index in [0,1]:
                for seed in tune_seeds:
                    rid=f'{prefix}_tune_{entry}_c{config_index}_s{seed}'
                    add(rid,prefix+'_tuning',method,role,seed,horizon,hpo+rescue,shortlist_index=config_index,
                        rank=rank,max_rank=2*rank,model=model,targets=targets)
                    tuning.append(rid)
            for seed in report_seeds:
                rid=f'{prefix}_screen_{entry}_s{seed}'
                add(rid,prefix+'_screen',method,role,seed,horizon,tuning,rank=rank,max_rank=2*rank,model=model,targets=targets)
                terminals.append(rid)
        return terminals
    screen=tournament('B',methods,512,1001,[1002,1003],[2001,2002,2003])
    add('selection_freeze','C_freeze',deps=screen,kind='registry_config_parser_analysis_freeze',sealed_until=['dev_audit','official_tests'])
    confirmation=[]
    for role in ROLES:
        for seed in range(3001,3006):
            rid=f'C_confirm_{role}_s{seed}'; confirmation.append(rid)
            add(rid,'C_confirmation',role=role,seed=seed,updates=512,deps=['selection_freeze'])
    # Explicit conditional coverage of matching contrasts and beating controls.
    controls=['L_GATE','I_GATE','T0_FIXED','T1_FIXED','T0_FIXED_ORTHO','T1_FIXED_ORTHO']
    controls += [m for m in methods if 'UNIFORM_GATE' in m or 'GLOBAL_RANDOM' in m or 'RESTART' in m]
    for method in controls:
        for seed in range(3001,3006):
            add(f'C_control_{method}_s{seed}','C_matched_controls',method,seed=seed,updates=512,deps=['selection_freeze'],
                conditional='required_interaction_pair_or_matched_triplet_or_beats_corresponding_adaptive',reuse_identical_branch=True)
    rank4=['L_FIXED','I_FIXED','ADALORA_REF']+[f'{i}_{s}' for i in ['L','I'] for s in ['RAW','GATE','DISP','ACT']]
    tournament('C_rank4',rank4,512,4101,[4102,4103],[4001,4002,4003],rank=4)
    calibrations=[]
    for seed in range(6101,6106):
        rid=f'D_calibration_L_FIXED_s{seed}'; calibrations.append(rid)
        add(rid,'D_calibration','L_FIXED',seed=seed,updates=512,deps=['selection_freeze'],quality_blind=True)
    for role in ROLES:
        for seed in range(5001,5004):
            add(f'D_export_{role}_s{seed}','D_export',role=role,seed=seed,updates=512,deps=['selection_freeze'],
                phases={'source':.75,'ordinary_lora_recovery':.25},standalone_accounting=True,
                exact_separate_process_reload=True)
        for repeat in [0,1]:
            for seed in range(6001,6004):
                add(f'D_compute_{role}_s{seed}_rep{repeat}','D_compute',role=role,seed=seed,
                    deps=calibrations+['selection_freeze'],
                    compute_cap={'median_gpu_seconds_from':calibrations,'max_overshoot':'one_update',
                                 'phase_progress':'charged_gpu_seconds'},hardware_repeat=repeat,
                    guaranteed_transition_and_recovery=True)
    tournament('E_7b',list(ROLES),1024,7101,[7102,7103],[7001,7002,7003],model='7B')
    tournament('E_hetero',list(ROLES),512,8101,[8102,8103],[8001,8002,8003],targets='attention_and_mlp')
    for method in ['L_GATE','I_GATE']:
        for seed in [2901,2902]:
            for p in [.20,.50,.75]:
                add(f'delete_{method}_s{seed}_p{p}','diagnostic_deletion',method,seed=seed,updates=512,deps=gates,
                    checkpoint_progress=p,slots=64,probe_examples=64,shadow_scores=['RAW','GATE','DISP','ACT'],
                    signed_damage=True,restore_rng_and_all_tensors=True)
    for init in ['L','I']:
        for seed in [2903,2904]:
            for branch in ['wide_continuation','wide_reset','exact_refactor_reset','truncated_uniform','truncated_global']:
                add(f'recovery_{init}_{seed}_{branch}','diagnostic_recovery',init+'_WIDE',seed=seed,updates=512,deps=gates,
                    branch=branch,source_fraction=.75,recovery_fraction=.25,selection_eligible=False)
    for probe in ['sigma_0.1','sigma_0.5','sigma_1.0','floor_0','floor_1','fixed_scale','changing_rank_normalization',
                  'effective_gradient_scale','upstream_unnormalized_penalty','signed_first_order','finite_difference_curvature','gauge_rescaling','interacting_deletions']:
        add(f'mechanism_{probe}','diagnostic_mechanism',deps=gates,probe=probe,selection_eligible=False,device='cpu')
    # Official evaluation jobs are explicit for all mandatory confirmed branches.
    trained=[r['run_id'] for r in rows if r['stage'] in {'B_screen','C_confirmation','C_matched_controls','C_rank4_screen','D_export','D_compute','E_7b_screen','E_hetero_screen'}]
    for parent in trained:
        for dataset in ['GSM8K_test','MATH500']:
            add(f'eval_{parent}_{dataset}','official_evaluation',deps=[parent,'selection_freeze'],dataset=dataset,
                parent_checkpoint=parent,max_new_tokens=2048,greedy=True,sealed=True)
    ids=[r['run_id'] for r in rows]
    assert len(ids)==len(set(ids))
    assert all(d in ids for r in rows for d in r['depends_on'])
    return {'schema':'rankstudy-full-plan-v3','selection_roles':ROLES,'method_count':len(methods),
            'scope':'All mandatory stages; unresolved roles/conditional rescue jobs remain visible.',
            'counts_by_stage':dict(collections.Counter(r['stage'] for r in rows)),
            'runs':rows,'hardware':{'actual_gpus':4,'model':'RTX PRO 6000 Blackwell Max-Q'},
            'selection_policy':'dev_tune only; all required families receive full-horizon screening',
            'search_cost_policy':'charge every trial, failed attempt, pilot and conditional rescue separately'}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=Path('artifacts/run_plan_v3.json')); args=parser.parse_args()
    methods=[m['id'] for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']]
    plan=build_plan(methods)
    args.out.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'total':len(plan['runs']),'stages':plan['counts_by_stage']}))


if __name__=='__main__': main()
