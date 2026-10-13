"""Guard validation-only finalist selection, ties, failures and immutable inputs."""
import tempfile
import unittest
from pathlib import Path
from experiments import init_common as c
from experiments import init_confirm_common as confirm


class ConfirmationSelectionTests(unittest.TestCase):
    def fixture(self, task, values):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);protocol=root/'protocol.md';protocol.write_text('predeclared seeds45-49')
        families=['accepted_anchor','accepted_gaussian_scale','challenger']
        candidates=[dict(candidate_id=f'{f}_{i}',family_id=f,learning_rate=lr,
                         spec={'kind':'random','config':{'mode':'accepted'} if f=='accepted_anchor'
                               else {'mode':'accepted_scaled','scale':1+i}})
                    for f in families for i,lr in enumerate((.0001,.001))]
        registry=dict(task=task,rank=2 if task=='aircraft' else 8,study_lane='centered',
                      fixed_lr=.001,learning_rates=[.0001,.001],candidates=candidates,
                      protocol_sha256=c.sha(protocol))
        c.write(root/'manifest.json',{'registry':registry})
        pilots=[]
        for candidate,value in zip(candidates,values):
            cid=candidate['candidate_id'];passing=value is not None
            pilot=dict(candidate_id=cid,status='pass' if passing else 'numerical_failure',
                       selection_value=value)
            if task=='extraction':
                directory=root/'pilots'/cid;result=dict(steps=48,seed=42,validation_nll=value)
                c.write(directory/'result.json',result)
                pilot.update(manifest_sha256=c.sha(root/'manifest.json'),
                             artifact_directory=str(directory.relative_to(root)),
                             result_sha256=c.sha(directory/'result.json'))
                c.write(directory/'receipt.json',pilot)
            else:
                directory=root/'fits'/cid
                fit=dict(candidate_id=cid,seed=42,status=pilot['status'])
                if task=='aircraft':
                    fit.update(stage='pilot',result=dict(epochs=10,evaluate_test=False,test=None,
                         best_validation=dict(macro_class_accuracy=value,cross_entropy=.1)))
                    pilot['selection_cross_entropy']=.1
                    filename='fit.json'
                else:
                    fit['evaluation']={task:{'validation':{'ndcg_at_10':value}}};filename='pilot.json'
                c.write(directory/filename,fit)
                pilot.update(fit_id=str(directory.relative_to(root)),pilot_sha256=c.sha(directory/filename))
            pilots.append(pilot)
        selected=c.select_candidates(registry,pilots,minimize=task=='extraction')
        selected['manifest_sha256']=c.sha(root/'manifest.json')
        c.write(root/'selections.json',selected)
        c.write(root/'pilots_audit.json',{'status':'pass','phase':'pilots',
            'input_sha256':{str(p.relative_to(root)):c.sha(p) for p in root.rglob('*.json')}})
        return root,protocol

    def test_each_task_direction_tie_and_failed_candidate(self):
        for task in ('scifact','nfcorpus','aircraft','extraction'):
            values=[.3,.4,.4,.5,.8,None] if task!='extraction' else [.8,.7,.7,.6,.3,None]
            root,protocol=self.fixture(task,values)
            result=confirm.derive(root,protocol)
            self.assertEqual(result['seeds'],[45,46,47,48,49])
            self.assertEqual(result['roles'],dict(validation_winner='challenger_0',accepted_anchor='accepted_anchor_1',
                                                 accepted_gaussian_scale='accepted_gaussian_scale_1'))
            self.assertFalse(result['test_used_for_selection'])

    def test_equal_scores_choose_lower_lr_then_candidate_id_and_deduplicate_roles(self):
        root,protocol=self.fixture('scifact',[.8,.8,.8,.8,.8,.8])
        result=confirm.derive(root,protocol)
        self.assertEqual(result['roles']['validation_winner'],'accepted_anchor_0')
        self.assertEqual(len(result['candidates']),2)
        path=root/'confirmation.json';c.write_once(path,result)
        self.assertEqual(confirm.load_registry(path,protocol,root),result)
        protocol.write_text('changed after selection')
        with self.assertRaises(AssertionError):confirm.load_registry(path,protocol,root)

    def test_test_evaluation_in_pilot_and_forged_selection_rejected(self):
        root,protocol=self.fixture('scifact',[.3,.4,.4,.5,.8,.7])
        path=root/'selections.json';selection=c.read(path)
        selection['selections']['challenger']['validation_tuned']='challenger_1';c.write(path,selection)
        with self.assertRaises(AssertionError):confirm.derive(root,protocol)
        root,protocol=self.fixture('scifact',[.3,.4,.4,.5,.8,.7])
        path=root/'fits/challenger_0/pilot.json';fit=c.read(path);fit['evaluation']['scifact']['test']={};c.write(path,fit)
        selection=c.read(root/'selections.json')
        next(p for p in selection['pilots'] if p['candidate_id']=='challenger_0')['pilot_sha256']=c.sha(path)
        c.write(root/'selections.json',selection)
        with self.assertRaises(AssertionError):confirm.derive(root,protocol)


if __name__=='__main__':unittest.main()
