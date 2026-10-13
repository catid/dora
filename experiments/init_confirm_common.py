"""Freeze fresh-seed finalists using only the registered pilot validation grid."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from experiments import init_common as common


SEEDS=[45,46,47,48,49]
ROLES=('validation_winner','accepted_anchor','accepted_gaussian_scale')


def derive(study_root,protocol,task=None):
    root=Path(study_root)
    manifest=common.read(root/'manifest.json')
    registry=manifest['registry']
    task=task or registry['task']
    assert task==registry['task'] and registry.get('study_lane','centered')=='centered'
    assert registry['protocol_sha256']==common.sha(protocol)
    selection=common.read(root/'selections.json')
    pilot_audit=common.read(root/'pilots_audit.json')
    assert pilot_audit['status']=='pass' and pilot_audit['phase']=='pilots'
    # Verify the immutable pilot evidence below. Retrieval final scoring
    # legitimately extends result.json while its pilot.json remains frozen.
    assert selection.get('test_used_for_selection') is False
    assert selection['manifest_sha256']==common.sha(root/'manifest.json')
    assert selection['registry_sha256']==common.digest_json(registry)
    candidates={c['candidate_id']:c for c in registry['candidates']}
    pilots=selection['pilots']
    assert len(pilots)==len(candidates) and {p['candidate_id'] for p in pilots}==set(candidates)
    bindings={}
    successful={}
    for pilot in pilots:
        cid=pilot['candidate_id']
        if task=='extraction':
            path=root/'pilots'/cid/'receipt.json'
            actual=common.read(path)
            assert actual==pilot
            assert actual['manifest_sha256']==common.sha(root/'manifest.json')
            if actual['status']=='pass':
                result_path=root/actual['artifact_directory']/'result.json'
                assert common.sha(result_path)==actual['result_sha256']
                result=common.read(result_path)
                assert result['steps']==48 and result['seed']==42 and 'test' not in result
                assert result['validation_nll']==actual['selection_value']
                bindings[str(result_path.relative_to(root))]=common.sha(result_path)
                secondary=0.
        else:
            path=root/pilot['fit_id']/('fit.json' if task=='aircraft' else 'pilot.json')
            assert common.sha(path)==pilot['pilot_sha256']
            actual=common.read(path)
            assert actual['candidate_id']==cid and actual['seed']==42
            assert actual['status']==pilot['status']
            if actual['status']=='pass':
                if task=='aircraft':
                    assert actual['stage']=='pilot' and actual['result']['epochs']==10
                    assert not actual['result']['evaluate_test'] and actual['result']['test'] is None
                    val=actual['result']['best_validation']
                    assert val['macro_class_accuracy']==pilot['selection_value']
                    assert val['cross_entropy']==pilot['selection_cross_entropy']
                    secondary=val['cross_entropy']
                else:
                    assert 'test' not in actual['evaluation'][task]
                    assert actual['evaluation'][task]['validation']['ndcg_at_10']==pilot['selection_value']
                    secondary=0.
        bindings[str(path.relative_to(root))]=common.sha(path)
        if actual['status']=='pass':
            assert math.isfinite(pilot['selection_value']) and math.isfinite(secondary)
            successful[cid]=(pilot['selection_value'],secondary)
        else:
            assert actual['status'] in ('failed_numerically','numerical_failure')
    assert successful, 'No successful pilot to confirm'
    for name,digest in {'manifest.json':common.sha(root/'manifest.json'),
                        'selections.json':common.sha(root/'selections.json'), **bindings}.items():
        assert pilot_audit['input_sha256'].get(name)==digest, f'Pilot is not bound by passing audit: {name}'

    def key(cid):
        primary,secondary=successful[cid]
        return (primary if task=='extraction' else -primary,secondary,candidates[cid]['learning_rate'],cid)

    roles={'validation_winner':min(successful,key=key)}
    for family in ROLES[1:]:
        eligible=[cid for cid in successful if candidates[cid]['family_id']==family]
        assert eligible, f'Required reference family failed: {family}'
        roles[family]=min(eligible,key=key)
        assert selection['selections'][family]['validation_tuned']==roles[family]
    for family, lanes in selection['selections'].items():
        for lane,cid in lanes.items():
            eligible=[key_ for key_ in successful if candidates[key_]['family_id']==family
                      and (lane!='fixed_original_lr' or candidates[key_]['learning_rate']==registry['fixed_lr'])]
            assert cid==(min(eligible,key=key) if eligible else None), (family,lane)
    # IDs can be shared between roles. Do not count the reused fit as a replicate.
    chosen=[candidates[cid] for cid in dict.fromkeys(roles.values())]
    return dict(schema='dora-init-confirmation-v1',task=task,rank=registry['rank'],
                study_lane='centered',seeds=SEEDS,roles=roles,candidates=chosen,
                control_id=roles['accepted_anchor'],
                fixed_lr=registry['fixed_lr'],learning_rates=registry['learning_rates'],
                source_study_root=str(root.resolve()),protocol_sha256=common.sha(protocol),
                parent_manifest_sha256=common.sha(root/'manifest.json'),
                parent_registry_sha256=common.digest_json(registry),
                parent_selections_sha256=common.sha(root/'selections.json'),
                parent_pilots_audit_sha256=common.sha(root/'pilots_audit.json'),
                pilot_evidence_sha256=bindings,test_used_for_selection=False,
                winner_rule='All successful centered pilots; validation NLL min for extraction, nDCG/macro max otherwise; Aircraft CE min secondary; lower LR then candidate ID.',
                builder_source_sha256=common.sha(__file__))


def load_registry(path,protocol,study_root,task=None):
    result=common.read(path)
    assert result==derive(study_root,protocol,task), 'Confirmation registry or validation provenance changed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study-root','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--task',choices=('scifact','nfcorpus','aircraft','extraction'))
    args=parser.parse_args()
    registry=derive(args.study_root,args.protocol,args.task)
    common.write_once(args.output,registry)
    print({key:registry[key] for key in ('task','roles','seeds')})


if __name__=='__main__':main()
