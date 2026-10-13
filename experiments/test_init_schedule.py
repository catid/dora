"""Scheduler DAG and fake-process lifecycle checks; never launches model jobs."""
import copy
import json
import tempfile
import unittest
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock
from experiments import init_schedule as scheduler
from experiments import init_common as common


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def make(self,jobs):
        self.plan=self.root/'schedule.json'
        common.write(self.plan,{'root':str(self.root),'jobs':[dict(job_id=name,command=['fake',name],dependencies=deps,gpu=gpu,priority=0,source=str(self.root)) for name,deps,gpu in jobs]})

    def execute(self,behavior=None):
        behavior=behavior or {};events=[]
        class Process:
            counter=200
            def __init__(self,argv,**kwargs):
                Process.counter+=1;self.pid=Process.counter;self.name=argv[1]
                self.polls=iter(behavior.get(self.name,[0]));self.last=0
                events.append(('start',self.name,kwargs['env']['CUDA_VISIBLE_DEVICES']))
            def poll(self):
                self.last=next(self.polls,self.last)
                if self.last is not None:events.append(('done',self.name,self.last))
                return self.last
        with mock.patch.object(scheduler.subprocess,'Popen',Process),mock.patch.object(scheduler.time,'sleep'),mock.patch('builtins.print'):
            scheduler.execute(self.plan,[0,2])
        return events

    def test_success_dependencies_and_complete_resume_without_duplicate_execution(self):
        self.make([('a',[],True),('b',['a'],False),('c',['b'],True)])
        events=self.execute();starts=[e[1] for e in events if e[0]=='start']
        self.assertEqual(starts,['a','b','c'])
        self.assertEqual(self.execute(),[])
        self.assertEqual(common.read(self.root/'scheduler_state.json')['status'],'pass')

    def test_partial_pass_resume_runs_only_remaining_jobs(self):
        self.make([('a',[],True),('b',['a'],True)])
        common.write(self.root/'scheduler_state.json',{'plan_sha256':common.sha(self.plan),'jobs':{'a':{'status':'pass'}}})
        self.assertEqual([e[1] for e in self.execute() if e[0]=='start'],['b'])

    def test_failure_stops_new_jobs_and_allows_existing_job_to_finish(self):
        self.make([('fail',[],True),('ongoing',[],True),('late',[],True),('dependent',['fail'],False)])
        with self.assertRaises(SystemExit):self.execute({'fail':[1],'ongoing':[None,0]})
        state=common.read(self.root/'scheduler_state.json')
        self.assertEqual(set(state['jobs']),{'fail','ongoing'})
        self.assertEqual(state['jobs']['fail']['status'],'failed')
        self.assertEqual(state['jobs']['ongoing']['status'],'pass')
        with self.assertRaises(AssertionError):self.execute()

    def test_interrupted_or_changed_plan_requires_explicit_resolution(self):
        self.make([('a',[],True)])
        common.write(self.root/'scheduler_state.json',{'plan_sha256':common.sha(self.plan),'jobs':{'a':{'status':'running'}}})
        with self.assertRaises(AssertionError):self.execute()
        common.write(self.root/'scheduler_state.json',{'plan_sha256':'wrong','jobs':{}})
        with self.assertRaises(AssertionError):self.execute()

    def test_cycle_fails_loudly_and_gpu_assignments_are_distinct(self):
        self.make([('a',['b'],True),('b',['a'],True)])
        with self.assertRaises(RuntimeError):self.execute()
        (self.root/'scheduler_state.json').unlink()
        self.make([('a',[],True),('b',[],True)])
        starts=[e for e in self.execute({'a':[None,0],'b':[0]}) if e[0]=='start']
        self.assertEqual({e[2] for e in starts},{'0','2'})

    def test_adopt_real_child_of_stopped_parent_retains_exit_status(self):
        for exitcode in (0, 7):
            with self.subTest(exitcode=exitcode), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); child_file=root/'child.json'
                program=("import subprocess,sys,json,time; from pathlib import Path; "
                         "child=subprocess.Popen([sys.executable,'-c',"
                         "'import time; time.sleep(.5); raise SystemExit('+sys.argv[2]+')']); "
                         "Path(sys.argv[1]).write_text(json.dumps(child.pid)); time.sleep(30)")
                parent=subprocess.Popen([sys.executable,'-c',program,str(child_file),str(exitcode)])
                try:
                    deadline=time.monotonic()+5
                    while not child_file.exists():
                        self.assertLess(time.monotonic(),deadline);time.sleep(.01)
                    os.kill(parent.pid,signal.SIGSTOP)
                    while scheduler.process_state(parent.pid)['state']!='T':time.sleep(.01)
                    child_pid=json.loads(child_file.read_text())
                    identity=scheduler.process_state(child_pid)
                    self.assertEqual(identity['parent_pid'],parent.pid)
                    plan=root/'schedule.json'
                    common.write(plan,{'root':str(root),'jobs':[dict(job_id='adopted',command=['unused'],dependencies=[],gpu=True,priority=0,source=str(root))]})
                    common.write(root/'scheduler_state.json',{'plan_sha256':common.sha(plan),'jobs':{
                        'adopted':{'status':'running','pid':child_pid,'gpu':0,'_monotonic_start':time.monotonic()}}})
                    adoption=root/'adoption.json'
                    common.write(adoption,{'new_plan_sha256':common.sha(plan),
                                 'original_scheduler':scheduler.process_state(parent.pid),'processes':{'adopted':identity}})
                    with mock.patch('builtins.print'):
                        if exitcode:
                            with self.assertRaises(SystemExit):scheduler.execute(plan,[0],adoption)
                        else:scheduler.execute(plan,[0],adoption)
                    state=common.read(root/'scheduler_state.json')['jobs']['adopted']
                    self.assertEqual(state['returncode'],exitcode)
                    self.assertEqual(state['adopted_exit_evidence']['wait_status'],exitcode<<8)
                    self.assertEqual(state['status'],'failed' if exitcode else 'pass')
                    self.assertEqual(common.read(root/'adoption_complete.json')['status'],'pass')
                    self.assertEqual(parent.wait(timeout=5),-signal.SIGKILL)
                finally:
                    if parent.poll() is None:parent.kill();parent.wait(timeout=5)


if __name__=='__main__':unittest.main()
