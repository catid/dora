import hashlib
import json
import sys
import os
import subprocess
from pathlib import Path

import pytest
import torch

from rankstudy.v3.stage_execution import (PlannedInterruption,run_stage,attempt_costs,
                                         latest_checkpoint,complete_checkpoint)


def test_stage_checkpoint_resume_preserves_attempts_and_avoids_double_charge(tmp_path):
    script=tmp_path/'fake_native.py'
    script.write_text('''
import argparse,json,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--updates',type=int)
p.add_argument('--stop-after',type=int);p.add_argument('--resume',type=Path);a=p.parse_args()
step=a.stop_after or a.updates
a.out.mkdir(exist_ok=True)
ck=a.out/f'checkpoint_{step:04d}';ck.mkdir()
torch.save({'extra':{'charged_s':float(step)}},ck/'training_state.pt')
(ck/'manifest.json').write_text(json.dumps({'complete':True,'files':{'training_state.pt':hashlib.sha256((ck/'training_state.pt').read_bytes()).hexdigest()}}))
(a.out/'metadata.json').write_text(json.dumps({'status':'complete' if step==a.updates else 'interrupted_checkpoint',
 'updates_finished':step,'charged_gpu_seconds':float(step),'setup_s':.1,'evaluation_s':.2,
 'config':{'resume':str(a.resume) if a.resume else None}}))
''')
    root=tmp_path/'stage'
    cmd=[sys.executable,str(script),'--out',str(root),'--updates','8']
    with pytest.raises(PlannedInterruption):
        run_stage(root,cmd,8,interrupt_after=3)
    old=root/'attempts/attempt_0000/metadata.json'
    digest=hashlib.sha256(old.read_bytes()).hexdigest()
    ck,md,receipt=run_stage(root,cmd,8,resume=True)
    assert md['updates_finished']==8 and len(receipt['attempts'])==2
    assert hashlib.sha256(old.read_bytes()).hexdigest()==digest
    costs=attempt_costs(receipt['attempts'])
    assert [r['observed_training_gpu_seconds'] for r in costs]==[3.,5.]
    assert all(r['process_wall_s']>0 for r in costs)
    _,_,cached=run_stage(root,cmd,8,resume=True)
    assert cached['cached_physical_execution']
    assert len(list((root/'attempts').iterdir()))==2
    with pytest.raises(ValueError,match='changed'):
        run_stage(root,cmd+['--irrelevant'],8,resume=True)


def test_checkpoint_selection_rejects_partial_or_corrupted_writes(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir()
    for step in [2,4]:
        ck=attempt/f'checkpoint_{step:04d}';ck.mkdir()
        torch.save({'extra':{'charged_s':float(step)}},ck/'training_state.pt')
        (ck/'manifest.json').write_text(json.dumps({'complete':True,'files':{
            'training_state.pt':hashlib.sha256((ck/'training_state.pt').read_bytes()).hexdigest()}}))
    assert latest_checkpoint([attempt],3).name=='checkpoint_0002'
    (attempt/'checkpoint_0004/training_state.pt').write_bytes(b'partial')
    assert not complete_checkpoint(attempt/'checkpoint_0004')
    assert latest_checkpoint([attempt],8).name=='checkpoint_0002'


@pytest.mark.parametrize('script',['scripts/run_one_v3.py','scripts/run_pipeline_v3.py'])
def test_rejected_invocation_does_not_corrupt_an_existing_running_record(tmp_path,script):
    out=tmp_path/'existing';out.mkdir()
    record=out/'metadata.json';record.write_text('{"status":"running","belongs_to":"another invocation"}\n')
    original=record.read_bytes()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run([sys.executable,script,'--method','L_FIXED','--out',str(out)],env=env,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'gate is closed' in result.stderr
    assert record.read_bytes()==original
