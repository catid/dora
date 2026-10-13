"""Actual subprocess races against a source-bound fixed-temporary-file loader."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments import init_locked_launch as launcher


TASK='''from pathlib import Path
import json,os,time
def data(args,tokenizer):
    output=Path(args.output);active=output/'active'
    with active.open('x') as stream:stream.write(str(os.getpid()))
    try:
        manifest={'train':['a','b'],'validation':['c'],'test':['d']}
        path=output/'split_manifest.json';temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest,sort_keys=True))
        time.sleep(.12)
        if args.fail:raise ValueError('loader fixture failure')
        temporary.replace(path)
        return manifest,{'sha':__import__('hashlib').sha256(path.read_bytes()).hexdigest()}
    finally:active.unlink()
'''
RUNNER='''import argparse,json,os,time
from pathlib import Path
from experiments import extraction
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--barrier',action='store_true');p.add_argument('--fail',action='store_true');args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
if args.barrier:
    (args.output/('ready_'+str(os.getpid()))).write_text('ready')
    deadline=time.monotonic()+5
    while len(list(args.output.glob('ready_*')))<2:
        assert time.monotonic()<deadline
        time.sleep(.01)
records,provenance=extraction.data(args,None)
(args.output/('result_'+str(os.getpid())+'.json')).write_text(json.dumps([records,provenance]))
'''


class LockedLaunchTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';package=source/'experiments';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (package/'extraction.py').write_text(TASK)
        (package/'init_extraction.py').write_text(RUNNER)
        return source

    def command(self,root,source,name,*arguments,task_hash=None):
        return [sys.executable,str(Path(launcher.__file__).resolve()),'--source-root',str(source),
                '--module','experiments.init_extraction','--launcher-sha256',launcher.sha(launcher.__file__),
                '--runner-sha256',launcher.sha(source/'experiments/init_extraction.py'),
                '--task-sha256',task_hash or launcher.sha(source/'experiments/extraction.py'),
                '--receipt',str(root/(name+'.json')),'--','--output',str(root/'output'),*arguments]

    def test_two_real_subprocesses_serialize_only_data_and_replay_unchanged_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.fixture(root)
            before={str(p):launcher.sha(p) for p in source.rglob('*.py')}
            processes=[subprocess.Popen(self.command(root,source,name,'--barrier'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                       for name in ('one','two')]
            for process in processes:
                out,err=process.communicate(timeout=10);self.assertEqual(process.returncode,0,(out,err))
            receipts=[json.loads((root/(name+'.json')).read_text()) for name in ('one','two')]
            for receipt in receipts:
                self.assertEqual(receipt['status'],'pass');self.assertTrue(receipt['frozen_sources_unchanged'])
                self.assertEqual(len(receipt['data_calls']),1)
                self.assertEqual(receipt['data_calls'][0]['split_after_sha256'],launcher.sha(root/'output/split_manifest.json'))
            self.assertGreater(max(r['data_calls'][0]['lock_wait_seconds'] for r in receipts),.05)
            results=[json.loads(p.read_text()) for p in (root/'output').glob('result_*.json')]
            self.assertEqual(len(results),2);self.assertEqual(results[0],results[1])
            self.assertEqual(results[0][0],{'train':['a','b'],'validation':['c'],'test':['d']})
            self.assertEqual(before,{str(p):launcher.sha(p) for p in source.rglob('*.py')})

    def test_hash_failure_precedes_data_and_loader_failure_releases_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.fixture(root)
            result=subprocess.run(self.command(root,source,'bad_hash',task_hash='0'*64),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            receipt=json.loads((root/'bad_hash.json').read_text());self.assertEqual(receipt['data_calls'],[])
            self.assertFalse((root/'output').exists())
            result=subprocess.run(self.command(root,source,'failed','--fail'),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            receipt=json.loads((root/'failed.json').read_text());self.assertEqual(receipt['data_calls'][0]['status'],'failed')
            result=subprocess.run(self.command(root,source,'recovery'),capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads((root/'recovery.json').read_text());self.assertEqual(receipt['status'],'pass')


if __name__=='__main__':unittest.main()
