import copy
from pathlib import Path
import unittest

from experiments import init_extension_registry as registry
from experiments import init_extension_report as report


class ExtensionReportTests(unittest.TestCase):
    def test_aircraft_list_diagnostics_are_retained(self):
        row={'initialization':{'diagnostics':[{'name':'blocks.0.fc1','initializer':{
            'initializer_kind':'extension_centered','callback_metadata':{'calibration_role':'task_only'}}}]}}
        result=report.initialization_evidence(row,'aircraft')
        self.assertEqual(result['modules']['blocks.0.fc1']['callback_metadata'],{'calibration_role':'task_only'})

    def test_model_probe_metadata_is_complete_but_not_repeated_per_module(self):
        metadata={'gme':{'observations':{'baseline':{'var':1.}},'solve_mode':'paper_ridge_damped'},
                  'factor_cache_hit':True,'source_compute_seconds':12.,'actual_cache_or_compute_seconds':.2}
        row={'initialization':{'initializer_seconds':.3},'initial_diagnostics':{
            name:{'initializer':{'initializer_kind':'extension_warmstart','callback_metadata':copy.deepcopy(metadata)}}
            for name in ('layer0.q','layer0.v')}}
        result=report.initialization_evidence(row,'scifact')
        self.assertEqual(len(result['model_gme_metadata']),1);self.assertEqual(len(result['modules']),2)
        self.assertEqual(next(iter(result['model_gme_metadata'].values())),metadata)
        self.assertEqual(result['initializer_seconds'],.3)

    def test_registry_geometry_and_complete_candidate_inventory(self):
        protocol=Path(registry.__file__).with_name('initialization_extension_protocol.md')
        value=registry.build('scifact',protocol)
        report.validate_registry(value)
        broken=copy.deepcopy(value);broken['family_definitions'][2]['initialization_lane']='warmstart'
        with self.assertRaises(AssertionError):report.validate_registry(broken)
        broken=copy.deepcopy(value);broken['candidates'].pop()
        with self.assertRaises(AssertionError):report.validate_registry(broken)

    def test_failed_gaussian_comparator_remains_unavailable_and_geometry_stays_explicit(self):
        fits={};cells={}
        for family,geometry in [('accepted_anchor','centered'),('accepted_gaussian_scale','centered'),('gme_paper','warmstart')]:
            keys=[]
            for seed in (42,43,44):
                key=f'{family}/seed_{seed}';keys.append(key)
                fits[key]={'status':'numerical_failure' if family=='accepted_gaussian_scale' and seed==43 else 'pass',
                           'seed':seed,'fit_id':key,'scores':[.5,.7],'learning_rate':.001,'spec':{},
                           **{k:None for k in ['training_seconds','initializer_seconds','peak_memory_bytes','total_fit_wall_seconds','trainable_parameters']}}
            for lane in report.primary.LANES:
                cells[family+'/'+lane]={'family_id':family,'lane':lane,'candidate_id':family,'fit_keys':keys,
                    'initialization_lane':geometry,'status':'final_seed_failure' if family=='accepted_gaussian_scale' else 'pass'}
        data={'study_lane':'extension','dataset':'scifact','cells':cells,'fits':fits,'example_ids':['a','b'],'strata':None}
        result=report.compute({'tasks':{'extension/scifact':data}},replicates=50)
        comparisons=[r for r in result['comparisons'] if r['comparison']=='equally_tuned_gaussian_scale']
        self.assertEqual(len(comparisons),6)
        self.assertTrue(all(r['status']=='unavailable_failed_or_missing_fit' for r in comparisons))
        self.assertTrue(all(r['initialization_lane']=='warmstart' for r in result['all_family_results'] if r['family_id']=='gme_paper'))


if __name__=='__main__':unittest.main()
