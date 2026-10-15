"""Small, local subprocess stage runner with immutable attempts and checkpoints."""
from __future__ import annotations
import hashlib
import json
import subprocess
import time
from pathlib import Path
import torch

from .checkpoint import json_write


class PlannedInterruption(RuntimeError):
    pass


def complete_checkpoint(path):
    path=Path(path)
    try:
        record=json.loads((path/'manifest.json').read_text())
        return bool(record.get('complete')) and all(
            hashlib.sha256((path/name).read_bytes()).hexdigest()==digest
            for name,digest in record['files'].items())
    except (OSError,ValueError,KeyError):
        return False


def latest_checkpoint(attempt_dirs, maximum_step):
    candidates=[]
    for attempt in attempt_dirs:
        for path in attempt.glob('checkpoint_*'):
            try: step=int(path.name.split('_')[-1])
            except ValueError: continue
            if step<=maximum_step and complete_checkpoint(path):
                candidates.append((step,str(path)))
    return Path(max(candidates)[1]) if candidates else None


def owned_live_process(launch):
    """A stale JSON/lock never counts as a live worker."""
    try:
        pid=launch['pid']; proc=Path(f'/proc/{pid}')
        stat=(proc/'stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z': return False
        if launch.get('start_ticks') and launch['start_ticks']!=stat[19]: return False
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        return b'--out' in argv and str(launch['out']).encode() in argv
    except (OSError,KeyError,IndexError):
        return False


def run_stage(root, command, endpoint, resume=False, interrupt_after=None):
    """Run/resume in a new attempt directory; never overwrite failed evidence.

    Parent logical charging comes from the completed checkpoint trajectory.
    All attempts additionally retain measured process wall time and cumulative
    native counters, so failed/rolled-back work is visible separately.
    """
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    canonical=root/'command.json'
    if canonical.exists():
        if json.loads(canonical.read_text())!=command: raise ValueError('stage command changed')
        if not resume: raise FileExistsError('existing stage requires explicit resume')
    else:
        json_write(canonical,command)
    dirs=sorted((root/'attempts').glob('attempt_*')) if (root/'attempts').exists() else []
    attempts=[]
    for path in dirs:
        launch_path=path/'launcher.json'
        launch=json.loads(launch_path.read_text()) if launch_path.exists() else {}
        if owned_live_process(launch): raise RuntimeError(f'stage child is still live: {launch["pid"]}')
        mdpath=path/'metadata.json'
        md=json.loads(mdpath.read_text()) if mdpath.exists() else {}
        attempts.append({'path':str(path),'launcher':launch,'metadata_status':md.get('status'),
                         'charged_gpu_seconds':md.get('charged_gpu_seconds'),
                         'setup_s':md.get('setup_s'),'evaluation_s':md.get('evaluation_s')})
        end=path/f'checkpoint_{endpoint:04d}'
        if not end.exists() and md.get('config',{}).get('resume'):
            candidate=Path(md['config']['resume'])
            if candidate.name==f'checkpoint_{endpoint:04d}': end=candidate
        if md.get('status') in {'complete','interrupted_checkpoint'} and md.get('updates_finished')==endpoint and complete_checkpoint(end):
            return end,md,{'cached_physical_execution':True,'attempts':attempts,'selected_attempt':str(path)}
    checkpoint=latest_checkpoint(dirs,endpoint)
    # A full final checkpoint with missing terminal metadata needs evaluation
    # and a final receipt. Native resume from endpoint does exactly that.
    attempt=root/'attempts'/f'attempt_{len(dirs):04d}'
    attempt.mkdir(parents=True,exist_ok=False)
    actual=list(command); actual[actual.index('--out')+1]=str(attempt)
    if checkpoint: actual+=['--resume',str(checkpoint)]
    if interrupt_after is not None:
        if not 0<interrupt_after<endpoint: raise ValueError('invalid diagnostic interruption boundary')
        if '--stop-after' in actual:
            actual[actual.index('--stop-after')+1]=str(interrupt_after)
        else: actual+=['--stop-after',str(interrupt_after)]
    wall_start=time.perf_counter(); utc_start=time.time()
    resume_charge=0.
    if checkpoint:
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        resume_charge=state['extra']['charged_s']
        if (checkpoint/'accounting.json').exists():
            resume_charge+=json.loads((checkpoint/'accounting.json').read_text())['checkpoint_write_s']
    with (attempt/'stdout.log').open('x') as log:
        proc=subprocess.Popen(actual,stdout=log,stderr=subprocess.STDOUT)
        try: ticks=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19]
        except OSError: ticks=None
        launch={'pid':proc.pid,'start_ticks':ticks,'out':str(attempt),'command':actual,
                'status':'running','start_unix_s':utc_start,'resume_checkpoint':str(checkpoint) if checkpoint else None,
                'resume_logical_charge_s':resume_charge}
        json_write(attempt/'launcher.json',launch)
        code=proc.wait()
    launch.update(returncode=code,status='complete' if code==0 else 'failed',process_wall_s=time.perf_counter()-wall_start)
    json_write(attempt/'launcher.json',launch)
    if code: raise RuntimeError(f'stage failed (exit {code}); evidence retained at {attempt}')
    md=json.loads((attempt/'metadata.json').read_text())
    attempts.append({'path':str(attempt),'launcher':launch,'metadata_status':md['status'],
                     'charged_gpu_seconds':md.get('charged_gpu_seconds'),
                     'setup_s':md.get('setup_s'),'evaluation_s':md.get('evaluation_s')})
    if md.get('updates_finished')!=endpoint:
        if interrupt_after is not None and md['status']=='interrupted_checkpoint':
            raise PlannedInterruption(f'checkpoint interruption at {attempt}')
        raise RuntimeError('stage did not reach its intended endpoint')
    end=attempt/f'checkpoint_{endpoint:04d}'
    # Evaluation-only resume may use a previously written final checkpoint.
    if not end.exists() and checkpoint and int(checkpoint.name.split('_')[-1])==endpoint:
        end=checkpoint
    if not complete_checkpoint(end): raise RuntimeError('stage endpoint checkpoint incomplete')
    result={'cached_physical_execution':False,'attempts':attempts,'selected_attempt':str(attempt)}
    json_write(root/'completion.json',{'checkpoint':str(end),'metadata':str(attempt/'metadata.json'),**result})
    return end,md,result


def attempt_costs(attempts):
    """No double charge of checkpoint history; rollback work stays visible."""
    rows=[]
    for item in attempts:
        path=Path(item['path']); launch=item['launcher']
        md=json.loads((path/'metadata.json').read_text()) if (path/'metadata.json').exists() else {}
        end_charge=md.get('charged_gpu_seconds')
        if end_charge is None and (path/'metrics.jsonl').exists():
            last=None
            for line in (path/'metrics.jsonl').read_text().splitlines():
                try: last=json.loads(line)
                except ValueError: break
            if last:
                end_charge=last['charged_s']
                ck=path/f'checkpoint_{last["step"]:04d}'/'accounting.json'
                if ck.exists(): end_charge+=json.loads(ck.read_text())['checkpoint_write_s']
        measured=max(0.,(end_charge or 0.)-launch.get('resume_logical_charge_s',0.))
        rows.append({'attempt':str(path),'observed_training_gpu_seconds':measured,
                     'setup_s':md.get('setup_s',0.),'evaluation_s':md.get('evaluation_s',0.),
                     'process_wall_s':launch.get('process_wall_s'),
                     'unmeasured_inflight_failure':launch.get('returncode')!=0})
    return rows
