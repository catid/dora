"""Persistent dependency scheduler for frozen initializer-study task shards.

Every subprocess uses argv (no shell), one physical GPU, an immutable registry,
and an isolated source snapshot. Failed jobs stop new scheduling; existing fits
finish and remain available for diagnosis. No automatic retries or substitutions.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

from experiments import init_common as common


def chunks(values, size):
    return [values[start:start+size] for start in range(0, len(values), size)]


def build_plan(root, source, registry_root, protocol, gate, aircraft_gate,
               tasks=('scifact','nfcorpus','extraction','aircraft'),
               lanes=('centered','warmstart','quantized')):
    jobs = []
    for lane in lanes:
        for task in tasks:
            registry_path = registry_root/lane/f'{task}.json'
            registry = common.load_registry(registry_path, protocol, task)
            prefix = f'{lane}_{task}'
            output = root/'runs'/lane/task
            module = 'retrieval' if task in ('scifact','nfcorpus') else task
            base = [sys.executable,'-m',f'experiments.init_{module}',
                    '--registry',str(registry_path),'--protocol',str(protocol),
                    '--output',str(output),'--gate-evidence',str(gate)]
            if module == 'retrieval':
                base += ['--task',task]
            if task == 'aircraft':
                base += ['--task-gate-evidence',str(aircraft_gate)]
            audit = [sys.executable,'-m',f'experiments.audit_init_{module}','--root',str(output)]

            def add(suffix, command, dependencies, gpu=True, priority=10):
                name = prefix+'_'+suffix
                jobs.append(dict(job_id=name, command=command, dependencies=dependencies,
                                 gpu=gpu, priority=priority, task=task, study_lane=lane,
                                 source=str(source), output=str(output),
                                 registry_sha256=common.sha(registry_path)))
                return name

            controls = add('controls',base+['--phase','controls'],[],priority=0 if lane=='centered' else 8)
            control_audit = add('controls_audit',audit+['--phase','controls'],[controls],
                                gpu=task=='extraction',priority=1)
            pilots = []
            # Keep each initializer's LR settings in the same process. This
            # allows bounded cache reuse while retaining many independent shards.
            per_shard = 48 if task != 'aircraft' else 32
            for index, group in enumerate(chunks([c['candidate_id'] for c in registry['candidates']],per_shard)):
                pilots.append(add(f'pilots_{index:03}',base+['--phase','pilots','--candidate-ids',*group],
                                  [control_audit],priority=3))
            selection = add('select',base+['--phase','select'],pilots,gpu=task=='extraction',priority=2)
            pilot_audit = add('pilots_audit',audit+['--phase','pilots'],[selection],
                              gpu=task=='extraction',priority=2)
            finals = []
            families = list(registry['candidate_counts'])
            # Quantized treatment finals also fit same-LR paired controls. One
            # shard serializes their shared checkpoint paths across families.
            for index, group in enumerate(chunks(families,len(families) if lane=='quantized' else 4)):
                finals.append(add(f'finals_{index:03}',base+['--phase','finals','--family-ids',*group],
                                  [pilot_audit],priority=3))
            if task != 'extraction':
                final_dependencies = [add('finalize',base+['--phase','finalize'],finals,gpu=False,priority=2)]
            else:
                final_dependencies = finals
            add('finals_audit',audit+['--phase','finals'],final_dependencies,
                gpu=task=='extraction',priority=2)
    return dict(schema='dora-init-schedule-v1',source=str(source),root=str(root),
                protocol_sha256=common.sha(protocol),gate_sha256=common.sha(gate),
                aircraft_gate_sha256=common.sha(aircraft_gate),jobs=jobs)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_state(pid):
    """Linux process identity and retained wait status, including stopped parents."""
    path = Path(f'/proc/{pid}/stat')
    fields = path.read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=fields[0], parent_pid=int(fields[1]),
                start_ticks=int(fields[19]), wait_status=int(fields[49]))


class AdoptedProcess:
    """Monitor a child of a stopped scheduler without killing its active fit."""
    def __init__(self, identity):
        self.identity = identity
        self.pid = identity['pid']
        self.exit_evidence = None

    def poll(self):
        if self.exit_evidence is not None:
            return os.waitstatus_to_exitcode(self.exit_evidence['wait_status'])
        current = process_state(self.pid)
        assert current['start_ticks'] == self.identity['start_ticks'], 'Adopted process PID was reused'
        assert current['parent_pid'] == self.identity['parent_pid'], 'Original scheduler must remain stopped until all children exit'
        if current['state'] != 'Z':
            return None
        self.exit_evidence = current
        return os.waitstatus_to_exitcode(current['wait_status'])


def execute(plan_path, gpus, adoption_path=None):
    plan = common.read(plan_path)
    root = Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _execute_locked(plan_path,plan,root,gpus,adoption_path)


def _execute_locked(plan_path,plan,root,gpus,adoption_path=None):
    ids=[job['job_id'] for job in plan['jobs']]
    assert len(ids)==len(set(ids)), 'Duplicate job IDs'
    assert all(set(job['dependencies'])<=set(ids) for job in plan['jobs']), 'Unknown dependency'
    state_path = root/'scheduler_state.json'
    state = common.read(state_path) if state_path.exists() else dict(
        plan_sha256=common.sha(plan_path),created_utc=now(),jobs={})
    assert state['plan_sha256']==common.sha(plan_path)
    assert set(state['jobs'])<=set(ids), 'Unknown job in scheduler state'
    running={name:row for name,row in state['jobs'].items() if row['status']=='running'}
    adoption=common.read(adoption_path) if adoption_path is not None else None
    if running:
        assert adoption is not None, 'Resolve interrupted jobs explicitly before resume'
        assert adoption['new_plan_sha256']==common.sha(plan_path)
        assert set(adoption['processes'])==set(running)
        original=process_state(adoption['original_scheduler']['pid'])
        assert original['start_ticks']==adoption['original_scheduler']['start_ticks'] and original['state']=='T'
        children={int(pid) for pid in Path(f"/proc/{original['pid']}/task/{original['pid']}/children").read_text().split()}
        assert children=={row['pid'] for row in adoption['processes'].values()}, 'Stopped parent has untracked children'
        for name,row in adoption['processes'].items():
            assert row['pid']==running[name]['pid'] and row['parent_pid']==original['pid']
        occupied=[row['gpu'] for row in running.values() if row['gpu'] is not None]
        assert len(occupied)==len(set(occupied)) and set(occupied)<=set(gpus)
    assert not any(x['status']=='failed' for x in state['jobs'].values()), 'Failed jobs require explicit diagnosis; no automatic retry'
    active = {name:(AdoptedProcess(adoption['processes'][name]),io.StringIO(),row['gpu'])
              for name,row in running.items()}
    adoption_released = not bool(running)
    failed = False
    logs = root/'job_logs';logs.mkdir(exist_ok=True)
    common.write(state_path,state)
    while True:
        for jid,(process,stream,gpu) in list(active.items()):
            code=process.poll()
            if code is None:
                continue
            stream.close()
            row=state['jobs'][jid]
            row.update(status='pass' if code==0 else 'failed',finished_utc=now(),returncode=code,
                       elapsed_seconds=time.monotonic()-row.pop('_monotonic_start'))
            if isinstance(process,AdoptedProcess):
                row['adopted_exit_evidence']=process.exit_evidence
            del active[jid]
            failed |= code != 0
            print(json.dumps({'event':'finished','job_id':jid,'gpu':gpu,'returncode':code}),flush=True)
            common.write(state_path,state)
        if not adoption_released and not any(isinstance(p,AdoptedProcess) for p,_,_ in active.values()):
            # Every original child has an independently observed zombie wait
            # status in the persisted state, so the stopped old parent can exit.
            original=process_state(adoption['original_scheduler']['pid'])
            assert original['start_ticks']==adoption['original_scheduler']['start_ticks'] and original['state']=='T'
            os.kill(original['pid'],signal.SIGKILL)
            adoption_released=True
            common.write_once(root/'adoption_complete.json',dict(status='pass',adoption_sha256=common.sha(adoption_path),
                               finished_utc=now(),jobs={name:state['jobs'][name]['adopted_exit_evidence'] for name in running}))
        if not active and (failed or len(state['jobs'])==len(plan['jobs'])):
            break
        if not failed:
            pool_path=root/'gpu_pool.json'
            pool=common.read(pool_path) if pool_path.exists() else gpus
            assert pool and len(pool)==len(set(pool)) and set(pool)<=set(gpus), 'Invalid runtime GPU pool'
            occupied={gpu for _,_,gpu in active.values() if gpu is not None}
            free=[gpu for gpu in pool if gpu not in occupied]
            cpu_available=sum(gpu is None for _,_,gpu in active.values())<2
            ready=[j for j in plan['jobs'] if j['job_id'] not in state['jobs']
                   and all(state['jobs'].get(dep,{}).get('status')=='pass' for dep in j['dependencies'])]
            ready.sort(key=lambda j:(j['priority'], plan['jobs'].index(j)))
            for job in ready:
                if job['gpu']:
                    if not free:continue
                    gpu=free.pop(0)
                else:
                    if not cpu_available:continue
                    gpu=None;cpu_available=False
                jid=job['job_id']
                environment={**os.environ,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu),
                             'PYTHONPATH':job['source'],'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8',
                             'MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'}
                stream=(logs/(jid+'.log')).open('ab')
                process=subprocess.Popen(job['command'],cwd=job['source'],env=environment,
                                         stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[jid]=(process,stream,gpu)
                state['jobs'][jid]=dict(status='running',started_utc=now(),pid=process.pid,gpu=gpu,
                                       command=job['command'],cwd=job['source'],
                                       log_path=str(logs/(jid+'.log')),
                                       environment_overrides={k:environment[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                                       _monotonic_start=time.monotonic())
                print(json.dumps({'event':'started','job_id':jid,'gpu':gpu,'pid':process.pid}),flush=True)
                common.write(state_path,state)
        if not active and not failed:
            raise RuntimeError('Unresolved scheduler dependency or no enabled GPUs')
        time.sleep(2)
    state.update(status='failed' if failed else 'pass',finished_utc=now())
    common.write(state_path,state)
    if failed:
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    build=commands.add_parser('plan')
    for name in ('root','source','registry-root','protocol','gate','aircraft-gate'):
        build.add_argument('--'+name,type=Path,required=True)
    build.add_argument('--tasks',nargs='+',default=['scifact','nfcorpus','extraction','aircraft'])
    build.add_argument('--lanes',nargs='+',default=['centered','warmstart','quantized'])
    run=commands.add_parser('run');run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    run.add_argument('--adoption',type=Path)
    args=parser.parse_args()
    if args.mode=='plan':
        plan=build_plan(args.root.resolve(),args.source.resolve(),args.registry_root.resolve(),
                        args.protocol.resolve(),args.gate.resolve(),args.aircraft_gate.resolve(),args.tasks,args.lanes)
        common.write_once(args.root/'schedule.json',plan)
        print(json.dumps({'jobs':len(plan['jobs']),'plan':str(args.root/'schedule.json')}))
    else:
        assert args.gpus and len(set(args.gpus))==len(args.gpus)
        execute(args.plan,args.gpus,args.adoption)


if __name__=='__main__':
    main()
