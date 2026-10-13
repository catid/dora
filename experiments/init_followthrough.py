"""Wait for audited main jobs, then run preregistered fresh-seed confirmation.

This queue owns only the GPUs explicitly released by its parent scheduler. Its
source snapshot must preserve the parent's task sources byte-for-byte.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import init_common as common
from experiments import init_confirm_schedule as confirmation

QUEUE_LOCK = None
QUEUE_LOCK_OWNED = False


def main():
    global QUEUE_LOCK, QUEUE_LOCK_OWNED
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-scheduler','parent-runs','confirmation-root','scheduler-source'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    args=parser.parse_args()
    root=args.confirmation_root.resolve();source=root/'source_run'
    root.mkdir(parents=True,exist_ok=True)
    QUEUE_LOCK=(root/'followthrough.lock').open('a')
    fcntl.flock(QUEUE_LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
    QUEUE_LOCK_OWNED=True
    for module in (common,confirmation):
        assert Path(module.__file__).resolve().is_relative_to(source), 'Imports must use the frozen confirmation source'
    parent_plan_path=args.parent_scheduler/'schedule.json'
    parent_plan=common.read(parent_plan_path)
    parent_plan_sha=common.sha(parent_plan_path)
    parent_job_ids={job['job_id'] for job in parent_plan['jobs']}
    assert len(parent_job_ids)==len(parent_plan['jobs'])
    record_path=root/'followthrough.json'
    previous=common.read(record_path) if record_path.exists() else None
    assert previous is None or previous['status'] in ('waiting_for_main','failed'), 'Queue already started'
    record={'schema':'dora-init-followthrough-v1','status':'waiting_for_main',
            'pid':os.getpid(),
            'process_start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
            'parent_scheduler':str(args.parent_scheduler.resolve()),'gpus':args.gpus,
            'source_sha256':common.sha(__file__),
            'parent_plan_sha256':parent_plan_sha,
            'scheduler_sha256':common.sha(args.scheduler_source)}
    common.write(record_path,record)
    print(record,flush=True)
    while True:
        parent=common.read(args.parent_scheduler/'scheduler_state.json')
        assert common.sha(parent_plan_path)==parent_plan_sha==parent['plan_sha256']
        if parent.get('status')=='failed':
            record.update(status='failed',reason='Parent scheduler failed; no confirmation launched')
            common.write(record_path,record);raise SystemExit(1)
        if parent.get('status')=='pass':
            assert set(parent['jobs'])==parent_job_ids
            assert all(row['status']=='pass' for row in parent['jobs'].values())
            break
        time.sleep(20)
    snapshot=common.read(root/'execution_snapshot.json')
    assert common.sha(args.scheduler_source)==record['scheduler_sha256'], 'Scheduler changed while queue waited'
    for name,digest in snapshot['source_sha256'].items():assert common.sha(source/name)==digest,name
    jobs=[];plans=[]
    for task in ('scifact','nfcorpus','extraction','aircraft'):
        task_root=root/task
        plan=confirmation.build_plan(task_root,source,(args.parent_runs/task).resolve(),
                                     root/'inputs/protocol.md',root/'inputs/backend_gate.json')
        common.write_once(task_root/'schedule.json',plan)
        jobs.extend(plan['jobs']);plans.append(str(task_root/'schedule.json'))
    aggregate=root/'orchestration'
    plan={'schema':'dora-init-schedule-v1','source':str(source),'root':str(aggregate),
          'jobs':jobs,'component_plan_sha256':{name:common.sha(name) for name in plans},
          'parent_final_scheduler_sha256':common.sha(args.parent_scheduler/'scheduler_state.json')}
    common.write_once(aggregate/'schedule.json',plan)
    command=[sys.executable,str(args.scheduler_source.resolve()),'run','--plan',str(aggregate/'schedule.json'),
             '--gpus',*[str(gpu) for gpu in args.gpus]]
    record.update(status='running',command=command,plan_sha256=common.sha(aggregate/'schedule.json'))
    common.write(record_path,record)
    environment={**os.environ,'PYTHONPATH':str(source)}
    result=subprocess.run(command,cwd=source,env=environment)
    record.update(status='pass' if result.returncode==0 else 'failed',returncode=result.returncode)
    common.write(record_path,record)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if QUEUE_LOCK_OWNED:
            root=Path(QUEUE_LOCK.name).parent
            path=root/'followthrough.json'
            record=common.read(path) if path.exists() else {}
            record.update(status='failed',error_type=type(error).__name__,error=str(error),
                          failing_queue_source_sha256=common.sha(__file__))
            common.write(path,record)
        raise
