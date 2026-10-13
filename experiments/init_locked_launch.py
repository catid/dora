"""Source-bound launch guard for the frozen ViGGO loader's shared .tmp file.

Wraps only extraction.data with a process lock. Frozen model, data selection,
runner, audit and training sources remain byte-for-byte unchanged. Do not wrap
confirmation/extension runners that already acquire this same non-reentrant lock.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

MODULES=('experiments.init_extraction','experiments.audit_init_extraction',
         'experiments.audit_init_confirm_extraction')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_receipt(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def locked_data(original,record,save):
    @functools.wraps(original)
    def wrapped(args,*positional,**keywords):
        output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
        lock_path=output/'.split_manifest.lock';split=output/'split_manifest.json'
        call={'output':str(output),'lock':str(lock_path),'started_utc':now()}
        started=time.perf_counter()
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            acquired=time.perf_counter();call['lock_wait_seconds']=acquired-started
            call['split_before_sha256']=sha(split) if split.exists() else None
            try:
                result=original(args,*positional,**keywords)
                call['status']='pass'
                return result
            except BaseException as error:
                call['status']='failed';call['error_type']=type(error).__name__
                raise
            finally:
                call['data_seconds']=time.perf_counter()-acquired
                call['split_after_sha256']=sha(split) if split.exists() else None
                call['finished_utc']=now();record['data_calls'].append(call)
                fcntl.flock(lock,fcntl.LOCK_UN)
                save()
    return wrapped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--module',choices=MODULES,required=True)
    for name in ('launcher-sha256','runner-sha256','task-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('runner_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    source=args.source_root.resolve();receipt=args.receipt.resolve()
    if receipt.exists():raise FileExistsError(f'Use a distinct receipt per launch: {receipt}')
    runner=source/(args.module.replace('.','/')+'.py');task_file=source/'experiments/extraction.py'
    argv=args.runner_args[1:] if args.runner_args[:1]==['--'] else args.runner_args
    record={'schema':'dora-init-locked-launch-v1','started_utc':now(),'pid':os.getpid(),
        'source_root':str(source),'module':args.module,'runner_args':argv,'data_calls':[],
        'scope':'Only extraction.data serialized; frozen runner and task sources unchanged',
        'source_sha256':{'launcher':args.launcher_sha256,'runner':args.runner_sha256,'task':args.task_sha256}}
    save=lambda:write_receipt(receipt,record)
    original=None;task=None;old_argv=sys.argv
    try:
        for label,path,expected in [('launcher',Path(__file__).resolve(),args.launcher_sha256),
                                    ('runner',runner,args.runner_sha256),('task',task_file,args.task_sha256)]:
            if sha(path)!=expected:raise ValueError(f'{label} source hash mismatch: {path}')
        if any(name=='experiments' or name.startswith('experiments.') for name in sys.modules):
            raise RuntimeError('Launcher must start before experiment modules are imported')
        os.chdir(source)
        sys.path.insert(0,str(source))
        task=importlib.import_module('experiments.extraction')
        if Path(task.__file__).resolve()!=task_file.resolve():raise ValueError('Task imported outside frozen source')
        if Path(importlib.util.find_spec(args.module).origin).resolve()!=runner.resolve():
            raise ValueError('Runner resolved outside frozen source')
        original=task.data;task.data=locked_data(original,record,save)
        record['status']='running';save()
        sys.argv=[str(runner),*argv]
        runpy.run_module(args.module,run_name='__main__',alter_sys=True)
        record['status']='pass';record['returncode']=0
    except SystemExit as error:
        code=error.code if isinstance(error.code,int) else (0 if error.code is None else 1)
        record['status']='pass' if code==0 else 'failed';record['returncode']=code
        raise
    except BaseException as error:
        record['status']='failed';record['returncode']=1
        record['error_type']=type(error).__name__;record['error']=str(error)
        raise
    finally:
        if task is not None and original is not None:task.data=original
        sys.argv=old_argv
        record['finished_utc']=now()
        record['frozen_sources_unchanged']=all(path.exists() and sha(path)==expected for path,expected in
            [(runner,args.runner_sha256),(task_file,args.task_sha256)])
        if record['status']=='pass' and not record['frozen_sources_unchanged']:
            record['status']='failed';record['returncode']=1;record['error']='Frozen source changed during execution'
            save()
            raise RuntimeError('Frozen source changed during execution')
        save()


if __name__=='__main__':main()
