"""Portable replay exercises absent host inputs and rejects integrity failures."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments import init_portable as portable
from experiments import init_report as primary


class PortableTests(unittest.TestCase):
    def fixture(self,base,extension=False):
        task=base/'host_task';task.mkdir();source=base/'host_source';(source/'experiments').mkdir(parents=True)
        names=['__init__','init_report','change_report']
        if extension:names+=['init_extension_report','init_extension_registry','init_extension','init_common']
        evidence=primary.Evidence()
        for name in names:
            original=Path(__file__).with_name(name+'.py');dest=source/'experiments'/original.name;shutil.copyfile(original,dest);evidence.add(dest)
        lane='extension' if extension else 'centered';cid='anchor';family='accepted_anchor'
        spec={'kind':'random','config':{'mode':'accepted'}}
        candidate={'candidate_id':cid,'family_id':family,'parent_family':'accepted','spec':spec,'learning_rate':.001}
        registry={'task':'scifact','candidates':[candidate],'fixed_lr':.001}
        manifest={'registry':registry,'protocol_sha256':primary.sha(Path(__file__))};primary.write(task/'manifest.json',manifest)
        selected={'manifest_sha256':primary.sha(task/'manifest.json'),'registry_sha256':primary.digest(registry),'test_used_for_selection':False,
            'pilots':[{'candidate_id':cid,'status':'pass','selection_value':1.}],
            'selections':{family:{name:cid for name in primary.LANES}}}
        primary.write(task/'selections.json',selected);primary.write(task/'scifact_splits.json',{'test':['q']});primary.write(task/'scifact_qrels.json',{'q':{'0':1}})
        fits={};metrics={'ndcg_at_10':1.,'mrr_at_10':1.,'recall_at_10':1.}
        for seed in primary.SEEDS:
            key=f'{cid}/seed_{seed}';fit_id='fits/'+key;dest=task/fit_id
            primary.write(dest/'result.json',{'fit_id':fit_id,'candidate_id':cid,'seed':seed,'status':'pass','evaluation':{'scifact':{'test':metrics}}})
            primary.write(dest/'scifact/test_per_query.json',[{'query_id':'q','top_10':[str(i) for i in range(10)],'scores':list(range(10,0,-1)),**metrics}])
            fits[key]={'candidate_id':cid,'seed':seed,'fit_id':str(dest),'status':'pass','learning_rate':.001,'spec':spec,'scores':[1.], 'metrics':metrics,
                **{k:None for k in ['training_seconds','initializer_seconds','peak_memory_bytes','total_fit_wall_seconds','trainable_parameters']}}
        tensor=task/'checkpoint.pt';tensor.write_bytes(b'tensor payload intentionally never loaded')
        for p in task.rglob('*'):
            if p.is_file():evidence.add(p)
        cells={f'{family}/{name}':{'family_id':family,'lane':name,'candidate_id':cid,'status':'pass','fit_keys':list(fits),'initialization_lane':'centered'} for name in primary.LANES}
        data={'study_lane':lane,'training_task':'scifact','dataset':'scifact','root':str(task),'audit':{'sha256':'0'*64},'selections':selected['selections'],
            'cells':cells,'fits':fits,'example_ids':['q'],'strata':None,'fixed_lr':.001,'family_definitions':{family:{'initialization_lane':'centered'}}}
        payload={'schema':'dora-init-extension-report-v1' if extension else 'dora-init-report-v1','fixture':True,'bootstrap_replicates':10,'tasks':{lane+'/scifact':data},
            'all_candidates':[{'study_lane':lane,'task':'scifact',**candidate,'pilot':{'status':'pass'}}], 'evidence':evidence.files}
        reporter=importlib.import_module('experiments.init_extension_report') if extension else primary
        report=base/'host_report';reporter.render(payload,reporter.compute(payload,10),report)
        return task,source,report,payload

    def test_primary_and_extension_replay_without_original_host_files(self):
        for extension in (False,True):
            with self.subTest(extension=extension),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);task,source,report,payload=self.fixture(root,extension)
                receipt=portable.bundle(report,source,root/'bundle',part_bytes=1000)
                self.assertEqual(receipt['counts']['excluded_original_paths'],1)
                task.rename(root/'unavailable_task');source.rename(root/'unavailable_source');report.rename(root/'unavailable_report')
                result=portable.replay(root/'bundle',root/'rebuilt',root,parts_only=True)
                self.assertEqual(result['prediction_rows_rescored'],3)
                self.assertEqual(result['excluded_tensor_payloads_verified'],0)
                self.assertTrue(result['original_host_paths_denied'])
                self.assertEqual(portable.sha(root/'unavailable_report/report_inputs.json'),portable.sha(root/'rebuilt/report_inputs.json'))

    def test_corrupt_part_and_unsafe_archive_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,source,report,_=self.fixture(root)
            receipt=portable.bundle(report,source,root/'bundle',part_bytes=1000)
            part=root/'bundle'/receipt['archive_parts'][0]['path'];part.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'byte/hash'):portable.replay(root/'bundle',root/'rebuilt',root,parts_only=True)
            receipt['archive_path']='../outside.tar.gz';portable.write(root/'bundle/bundle_manifest.json',receipt)
            with self.assertRaisesRegex(ValueError,'Unsafe'):portable.replay(root/'bundle',root/'rebuilt',root)

    def test_unbound_dependency_and_corrupt_original_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,source,report,_=self.fixture(root)
            helper=source/'experiments/change_report.py';helper.write_text(helper.read_text()+'\n# unbound source change\n')
            with self.assertRaisesRegex(ValueError,'not bound'):portable.bundle(report,source,root/'bundle')
            (report/'summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Original report changed'):portable.bundle(report,source,root/'bundle')

    def test_raw_rescore_rejects_tampered_normalized_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,_,_,payload=self.fixture(root)
            next(iter(next(iter(payload['tasks'].values()))['fits'].values()))['scores']=[.5]
            with self.assertRaises(AssertionError):portable.rescore(payload,primary,primary.Evidence(),primary)

    def test_path_traversal_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'escape').symlink_to(root.parent)
            for name in ('/absolute','../parent','a/../../parent','escape/outside'):
                with self.assertRaises(ValueError):portable.local(root,name)


if __name__=='__main__':unittest.main()
