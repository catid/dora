import yaml
from pathlib import Path
from collections import Counter
from scripts.make_run_plan_v3 import build_plan, ROLES, trial_design


def plan():
    methods=[m['id'] for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']]
    return build_plan(methods)


def test_all_41_have_complete_equal_hpo_tuning_and_fresh_screen():
    p=plan()
    counts=Counter((r['method_id'],r['stage']) for r in p['runs'])
    methods={r['method_id'] for r in p['runs'] if r['stage']=='B_hpo'}
    assert len(methods)==41
    for m in methods:
        assert counts[m,'B_hpo']==12
        assert counts[m,'B_tuning']==4
        assert counts[m,'B_screen']==3
        assert counts[m,'B_rescue']==2


def test_guaranteed_family_coverage_and_rank4_scores():
    p=plan()
    assert len(ROLES)==11
    for stage,n in [('C_confirmation',5),('D_export',3),('D_compute',6),('E_7b_screen',3),('E_hetero_screen',3)]:
        counts=Counter(r['selection_role'] for r in p['runs'] if r['stage']==stage)
        assert counts=={role:n for role in ROLES}
    r4={r['method_id'] for r in p['runs'] if r['stage']=='C_rank4_screen'}
    assert {f'{i}_{s}' for i in ['L','I'] for s in ['RAW','GATE','DISP','ACT']} <= r4
    for prefix in ['E_7b','E_hetero']:
        for role in ROLES:
            assert sum(r['stage']==prefix+'_hpo' and r['selection_role']==role for r in p['runs'])==12
            assert sum(r['stage']==prefix+'_tuning' and r['selection_role']==role for r in p['runs'])==4


def test_trial_design_fixed_before_results_and_reference_quota():
    for method in ['L_FIXED','I_FIXED','T0_GATE_ORTHO','ADALORA_REF']:
        trials=trial_design(method)
        assert len(trials)==12
        assert len({(t['lr'],t['scale'],t['penalty']) for t in trials})==12
    ref=trial_design('ADALORA_REF')
    canonical=[t for t in ref if t['canonical_reference_replacement']]
    assert len(canonical)==1 and canonical[0]['lr']==8e-4 and canonical[0]['scale']==1. and canonical[0]['penalty']==.1


def test_dependency_graph_is_acyclic_and_selection_precedes_large_models():
    p=plan(); rows={r['run_id']:r for r in p['runs']}; visited=set(); stack=set()
    def walk(rid):
        if rid in stack: raise AssertionError('cycle')
        if rid in visited: return
        stack.add(rid)
        for dep in rows[rid]['depends_on']: walk(dep)
        stack.remove(rid); visited.add(rid)
    for rid in rows: walk(rid)
    for r in rows.values():
        if r['stage'] in ['E_7b_hpo','E_hetero_hpo']:
            assert 'selection_freeze' in r['depends_on']
