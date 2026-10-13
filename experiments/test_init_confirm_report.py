"""Fresh-seed report estimands, deduplication and failure visibility."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from experiments import init_confirm_report as report


class ConfirmationReportTests(unittest.TestCase):
    def test_five_seed_metric_and_strata_are_explicit(self):
        values=np.array([[1,0,0,0,0]]*5)
        np.testing.assert_array_equal(report.metric(values),[.2]*5)
        np.testing.assert_array_equal(report.metric(values,[0,1,1,1,1]),[.5]*5)
        with self.assertRaises(AssertionError):report.metric(values[:3])

    def test_identical_roles_and_constant_class_effect(self):
        values=np.array([[1,0,0,0,0]]*5);reference=np.zeros_like(values)
        same=report.contrast(values,values,[0,1,1,1,1],1000)
        self.assertEqual(same['ci95'],[0.,0.]);self.assertEqual(same['paired_seed_differences'],[0.]*5)
        changed=report.contrast(values,reference,[0,1,1,1,1],1000)
        self.assertEqual(changed['difference'],50.);self.assertEqual(changed['ci95'],[50.,50.])

    def test_five_seed_crossed_bootstrap_matches_independent_small_reference(self):
        values=np.arange(20,dtype=float).reshape(5,4)/20
        result=report.contrast(values,np.zeros_like(values),None,10)
        rng=np.random.default_rng(20261009);seeds=rng.integers(5,size=(10,5));examples=rng.choice(np.arange(4),size=(10,4))
        boot=[]
        for row in range(10):boot.append(np.mean([values[s,e] for s in seeds[row] for e in examples[row]])*100)
        np.testing.assert_allclose(result['ci95'],np.quantile(boot,[.025,.975]),rtol=0,atol=1e-12)

    def test_failed_seed_retained_and_shared_roles_use_one_candidate(self):
        candidate={'spec':{'kind':'random','config':{'mode':'accepted'}},'learning_rate':.001}
        roles={role:'a' for role in report.ROLES}
        fits={f'a/seed_{seed}':{'seed':seed,'status':'pass' if seed!=47 else 'numerical_failure','fit_id':f'fit/{seed}'} for seed in report.SEEDS}
        payload={'tasks':{'scifact':{'roles':roles,'candidates':{'a':candidate},'fits':fits}}}
        result=report.compute(payload)
        self.assertEqual(len(result['results']),3)
        self.assertEqual(len({fid for row in result['results'] for fid in row['fit_ids']}),5)
        for row in result['results']:
            self.assertEqual(row['failed_seeds'],[47]);self.assertNotIn('mean',row)
        self.assertTrue(all(row['status']=='unavailable_failed_confirmation_seed' for row in result['comparisons']))

    def test_confirmation_collection_checks_parent_selection_and_five_prediction_sets(self):
        # Synthetic schema-only fixture: no runtime or downstream quality claim.
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);parent=base/'parent';root=base/'confirmation';write=report.report.write;sha=report.report.sha
            families=['challenger','accepted_anchor','accepted_gaussian_scale']
            candidates=[{'candidate_id':family,'family_id':family,'learning_rate':.001,
                         'spec':{'kind':'random','config':{'mode':'accepted'}}} for family in families]
            registry={'task':'scifact','candidates':candidates,'fixed_lr':.001}
            write(parent/'manifest.json',{'registry':registry,'source_sha256':{}})
            pilots=[];bindings={}
            for candidate,value in zip(candidates,[.8,.5,.6]):
                cid=candidate['candidate_id'];pilot={'candidate_id':cid,'status':'pass','selection_value':value}
                path=parent/'fits'/cid/'pilot.json';write(path,pilot);bindings[str(path.relative_to(parent))]=sha(path);pilots.append(pilot)
            selected={'test_used_for_selection':False,'registry_sha256':report.report.digest(registry),'pilots':pilots,
                      'selections':{family:{lane:family for lane in report.report.LANES} for family in families}}
            write(parent/'selections.json',selected)
            write(parent/'pilots_audit.json',{'status':'pass','phase':'pilots','input_sha256':{
                  'manifest.json':sha(parent/'manifest.json'),'selections.json':sha(parent/'selections.json'),**bindings}})
            confirmed={'schema':'dora-init-confirmation-v1','task':'scifact','seeds':report.SEEDS,'study_lane':'centered',
                       'test_used_for_selection':False,'roles':dict(zip(report.ROLES,families)),
                       'candidates':candidates,'source_study_root':str(parent),
                       'parent_manifest_sha256':sha(parent/'manifest.json'),'parent_registry_sha256':report.report.digest(registry),
                       'parent_selections_sha256':sha(parent/'selections.json'),'parent_pilots_audit_sha256':sha(parent/'pilots_audit.json'),
                       'pilot_evidence_sha256':bindings}
            write(root/'manifest.json',{'registry':confirmed,'parent_manifest_sha256':sha(parent/'manifest.json'),
                                       'source_sha256':{},'configuration':{}})
            write(root/'task_audit.json',{'status':'pass','phase':'finals','input_sha256':{},'checkpoint_sha256':{}})
            write(root/'scifact_splits.json',{'test':['q']});write(root/'scifact_qrels.json',{'q':{'0':1}})
            ranking=[str(i) for i in range(10)];metrics=report.report.ranked_metrics(ranking,{'0':1});runs=[]
            for role,cid in confirmed['roles'].items():
                for seed in report.SEEDS:
                    fit_id=f'fits/{cid}/seed_{seed}';row={'candidate_id':cid,'seed':seed,'status':'pass','fit_id':fit_id,
                        'manifest_sha256':sha(root/'manifest.json'),'evaluation':{'scifact':{'test':metrics}}}
                    write(root/fit_id/'result.json',row)
                    write(root/fit_id/'scifact/test_per_query.json',[{'query_id':'q','top_10':ranking,'scores':list(range(10,0,-1)),**metrics}])
                    runs.append({**row,'role':role})
            write(root/'results.json',{'complete':True,'manifest_sha256':sha(root/'manifest.json'),'runs':runs})
            collected=report.collect(report.report.Evidence(),root,'scifact')
            self.assertEqual(len(collected['scifact']['fits']),15)
            result=report.compute({'tasks':collected})
            self.assertTrue(all(row['mean']==100. for row in result['results']))
            self.assertTrue(all(row['ci95']==[0.,0.] for row in result['comparisons']))
            write(root/'task_audit.json',{'status':'pass','phase':'controls'})
            with self.assertRaises(AssertionError):report.collect(report.report.Evidence(),root,'scifact')


if __name__=='__main__':unittest.main()
