"""Reporting safety checks independent of model training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments import init_report as report


class ReportTests(unittest.TestCase):
    def registry(self):
        candidates=[{'candidate_id':str(i),'family_id':'family','learning_rate':lr,'spec':{}}
                    for i,lr in enumerate([.1,.1,.2,.2])]
        registry={'candidates':candidates,'fixed_lr':.1}
        pilots=[{'candidate_id':str(i),'status':'pass','selection_value':value}
                for i,value in enumerate([.2,.3,.4,.1])]
        selected={'test_used_for_selection':False,'registry_sha256':report.digest(registry),
                  'pilots':pilots,'selections':{'family':{report.LANES[0]:'1',report.LANES[1]:'2'}}}
        return registry,selected

    def test_selection_is_recomputed_and_test_selection_rejected(self):
        registry,selected=self.registry()
        report.selections(registry,selected,'scifact')
        wrong=copy.deepcopy(selected);wrong['selections']['family'][report.LANES[0]]='0'
        with self.assertRaises(AssertionError):report.selections(registry,wrong,'scifact')
        wrong=copy.deepcopy(selected);wrong['test_used_for_selection']=True
        with self.assertRaises(AssertionError):report.selections(registry,wrong,'scifact')

    def test_selection_failure_and_aircraft_validation_tie(self):
        registry,selected=self.registry()
        for pilot in selected['pilots']:pilot.update(status='numerical_failure',selection_value=None)
        selected['selections']['family']={name:None for name in report.LANES}
        report.selections(registry,selected,'aircraft')
        for pilot,ce in zip(selected['pilots'],[.3,.2,.1,.4]):
            pilot.update(status='pass',selection_value=.5,selection_cross_entropy=ce)
        selected['selections']['family']={report.LANES[0]:'1',report.LANES[1]:'2'}
        report.selections(registry,selected,'aircraft')

    def test_extraction_schema_and_prediction_metrics(self):
        target={'dialogue_act':'verify_attribute','name':'Test','specifier':'release','exp_release_date':'2027'}
        gold=[{'id':'id','target':target,'utterance':'Test'}]
        row={'id':'id','gold':target,'utterance':'Test','text':json.dumps(target),'parsed':target,'exact_match':True}
        scores,metrics=report.extraction_scores([row],gold)
        self.assertEqual(scores,[1.]);self.assertEqual(metrics['schema_valid'],1.)
        row['exact_match']=False
        with self.assertRaises(AssertionError):report.extraction_scores([row],gold)

    def test_prediction_metrics_do_not_trust_stored_retrieval_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'fit/scifact/test_per_query.json';path.parent.mkdir(parents=True)
            ranking=[str(i) for i in range(10)];gold={'q':{'0':1}}
            metrics=report.ranked_metrics(ranking,gold['q'])
            path.write_text(json.dumps([{'query_id':'q','top_10':ranking,'scores':list(range(10,0,-1)),**metrics}]))
            row={'fit_id':'fit','evaluation':{'scifact':{'test':metrics}}}
            ids,strata,scores,_=report.rescore(report.Evidence(),root,'scifact','scifact',row,{'test':['q']},gold)
            self.assertEqual(ids,['q']);self.assertEqual(scores,[1.]);self.assertIsNone(strata)
            corrupt=json.loads(path.read_text());corrupt[0]['ndcg_at_10']=.5;path.write_text(json.dumps(corrupt))
            with self.assertRaises(AssertionError):report.rescore(report.Evidence(),root,'scifact','scifact',row,{'test':['q']},gold)

    def test_audit_requires_finals_and_matching_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifest.json').write_text('{}')
            (root/'input.json').write_text('{}')
            audit={'status':'pass','phase':'pilots','input_sha256':{'input.json':report.sha(root/'input.json')}}
            report.write(root/'task_audit.json',audit)
            with self.assertRaises(AssertionError):report.audit_root(report.Evidence(),root,{'source_sha256':{}})
            audit['phase']='finals';report.write(root/'task_audit.json',audit)
            report.audit_root(report.Evidence(),root,{'source_sha256':{}})
            (root/'input.json').write_text('{"changed":true}')
            with self.assertRaises(AssertionError):report.audit_root(report.Evidence(),root,{'source_sha256':{}})

    def test_bootstrap_pairs_seeds_and_class_stratifies(self):
        reference=np.zeros((3,5));candidate=np.array([[1,0,0,0,0]]*3);labels=[0,1,1,1,1]
        result=report.contrast(candidate,reference,labels,1000)
        self.assertEqual(result['difference'],50.)
        self.assertEqual(result['ci95'],[50.,50.])
        exact=report.contrast(candidate,candidate,labels,1000)
        self.assertEqual(exact['ci95'],[0.,0.])
        self.assertEqual(exact['paired_seed_differences'],[0.,0.,0.])

    def test_failed_final_seed_is_not_removed_for_summary(self):
        fits={f'a/seed_{seed}':{'seed':seed,'status':'pass' if seed!=43 else 'numerical_failure'} for seed in report.SEEDS}
        cells={f'accepted_anchor/{lane}':{'family_id':'accepted_anchor','lane':lane,'candidate_id':'a',
                                         'status':'final_seed_failure','fit_keys':list(fits)} for lane in report.LANES}
        data={'study_lane':'centered','dataset':'scifact','cells':cells,'fits':fits}
        result=report.compute({'tasks':{'centered/scifact':data}},1000)
        self.assertTrue(result['comparisons'])
        self.assertTrue(all(c['status']=='unavailable_failed_or_missing_fit' for c in result['comparisons']))
        self.assertEqual(len(result['all_family_results']),2)
        for row in result['all_family_results']:
            self.assertEqual(row['failed_seeds'],[43]);self.assertNotIn('mean',row)

    def test_display_list_is_declared_without_scores(self):
        self.assertEqual(len(report.REPRESENTATIVES),18)
        self.assertEqual(len(set(report.REPRESENTATIVES)),18)
        self.assertEqual(report.REPRESENTATIVES[:2],['accepted_anchor','accepted_gaussian_scale'])

    def test_parent_search_budget_includes_children_failures_and_counts_transfer_once(self):
        candidates=[{'study_lane':'centered','task':'nfcorpus','family_id':f,'parent_family':'p',
                     'spec':{'gain':g},'learning_rate':lr,'pilot':{'status':'pass' if g==1 else 'numerical_failure'}}
                    for f in ('a','b') for g in (1,2) for lr in (.1,.2,.3)]
        data={'study_lane':'centered','training_task':'nfcorpus','fixed_lr':.1,
              'candidate_counts':{'a':6,'b':6},'parent_candidate_counts':{'p':12}}
        payload={'tasks':{'nfcorpus':data,'transfer':copy.deepcopy(data)},'all_candidates':candidates}
        rows=report.search_budgets(payload);self.assertEqual(len(rows),3)
        parent=next(r for r in rows if r['scope']=='parent')
        self.assertEqual(parent['initializer_count'],4);self.assertEqual(parent['learning_rate_count'],3)
        self.assertEqual(parent['fixed_lr_candidate_count'],4);self.assertEqual(parent['failed_pilots'],6)
        data['candidate_counts']['a']=5
        with self.assertRaises(AssertionError):report.search_budgets(payload|{'tasks':{'nfcorpus':data}})


if __name__=='__main__':unittest.main()
