"""Build the finite separate extension; never modifies the frozen main registry."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments import init_common as common
from experiments import init_extension as extension

TASKS = {
    'scifact': {'rank':8,'fixed_lr':.001,'learning_rates':[.0001,.001]},
    'nfcorpus': {'rank':8,'fixed_lr':.0001,'learning_rates':[.00003,.0001,.0003,.001]},
    'aircraft': {'rank':2,'fixed_lr':.001,'learning_rates':[.00001,.0001,.0003,.001]},
    'extraction': {'rank':8,'fixed_lr':.0003,'learning_rates':[.0001,.0003]},
}


def build(task, protocol):
    settings=TASKS[task]
    families=[{'family_id':'accepted_anchor','parent_family':'accepted','kind':'random',
               'configs':[{'mode':'accepted'}],'source_default_index':0,'calibration_requirements':[]},
              {'family_id':'accepted_gaussian_scale','parent_family':'accepted','kind':'random',
               'configs':[{'mode':'accepted_scaled','scale':gain} for gain in (.25,.5,1.,2.)],
               'source_default_index':2,'calibration_requirements':[]}]
    families+=extension.extension_families()
    for family in families:
        family['initialization_lane']='warmstart' if family['kind']=='extension_warmstart' else 'centered'
    candidates=[];lookup={}
    for family in families:
        for index,config in enumerate(family['configs']):
            for lr_index,lr in enumerate(settings['learning_rates']):
                cid=f"{family['family_id']}__i{index}__lr{lr_index}"
                lookup[family['family_id'],index,lr]=cid
                candidates.append({'candidate_id':cid,'family_id':family['family_id'],
                    'parent_family':family['parent_family'],'initializer_index':index,
                    'initialization_lane':family['initialization_lane'],
                    'source_default_setting':index==family['source_default_index'],
                    'learning_rate':lr,'spec':{'kind':family['kind'],'config':config}})
    for candidate in candidates:
        candidate['comparison_control_id']=lookup['accepted_anchor',0,candidate['learning_rate']]
    return {'schema':'dora-init-extension-v1','study_lane':'extension','task':task,**settings,
        'seed':42,'seeds':[42,43,44],'protocol_sha256':common.sha(protocol),
        'control_id':lookup['accepted_anchor',0,settings['fixed_lr']],
        'family_definitions':families,'candidates':candidates,
        'candidate_counts':dict(Counter(c['family_id'] for c in candidates)),
        'parent_candidate_counts':dict(Counter(c['parent_family'] for c in candidates)),
        'registry_builder_sha256':common.sha(__file__),'backend_sha256':common.sha(extension.__file__),
        'calibration':{'samples':128,'seed':20261009,'batch_size':8,'split':'train',
                       'gme_accumulation_dtype':'float64','null_covariance_key':'corda_input_cov'},
        'independent_from_main':True,'test_used_for_candidate_definition':False}


def load_registry(path, protocol, task=None):
    registry=common.read(path)
    assert registry['schema']=='dora-init-extension-v1'
    if task is not None:assert registry['task']==task
    expected=build(registry['task'],protocol)
    assert registry==expected,'Extension registry differs from its frozen backend, builder, protocol or finite grid'
    return registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for task in TASKS:
        registry=build(task,args.protocol)
        common.write_once(args.output/f'{task}.json',registry)
        print(task,len(registry['candidates']),flush=True)


if __name__=='__main__':main()
