"""Schedule fresh-seed confirmation after a complete, audited pilot search."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from experiments import init_common as common
from experiments import init_confirm_common as confirmation


def build_plan(root, source, study_root, protocol, gate):
    registry = confirmation.derive(study_root, protocol)
    task = registry['task']
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / 'registry.json'
    common.write_once(registry_path, registry)
    output = root / 'run'
    module = 'retrieval' if task in ('scifact', 'nfcorpus') else task
    base = [sys.executable, '-m', f'experiments.init_confirm_{module}',
            '--registry', str(registry_path), '--protocol', str(protocol),
            '--study-root', str(study_root), '--output', str(output),
            '--gate-evidence', str(gate)]
    if module == 'retrieval':
        base += ['--task', task]
    audit_module = 'audit_init_confirm_extraction' if task == 'extraction' else 'audit_init_confirmation'
    audit = [sys.executable, '-m', f'experiments.{audit_module}', '--root', str(output)]
    if task == 'extraction':
        audit += ['--study-root', str(study_root)]
    jobs = []

    def add(suffix, command, dependencies, gpu=True, priority=1):
        jid = f'confirm_{task}_{suffix}'
        jobs.append(dict(job_id=jid, command=command, dependencies=dependencies,
                         gpu=gpu, priority=priority, task=task,
                         study_lane='confirmation', source=str(source),
                         output=str(output), registry_sha256=common.sha(registry_path)))
        return jid

    controls = add('controls', base + ['--phase', 'controls'], [])
    control_audit = add('controls_audit', audit + ['--phase', 'controls'],
                        [controls], gpu=task == 'extraction')
    # One shard per seed keeps each fit path exclusive. Controls have already
    # written the shared manifests, calibration and initial-reference evidence.
    fits = [add(f'seed_{seed}', base + ['--phase', 'finals' if task == 'extraction' else 'fits',
                                       '--seeds', str(seed)], [control_audit], priority=2)
            for seed in registry['seeds']]
    if task != 'extraction':
        fits = [add('finalize', base + ['--phase', 'finalize'], fits, gpu=False)]
    add('finals_audit', audit + ['--phase', 'finals'], fits, gpu=task == 'extraction')
    return dict(schema='dora-init-schedule-v1', source=str(source), root=str(root),
                study_root=str(study_root), protocol_sha256=common.sha(protocol),
                gate_sha256=common.sha(gate), registry_sha256=common.sha(registry_path), jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'study-root', 'protocol', 'gate'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(*(getattr(args, name).resolve()
                        for name in ('root', 'source', 'study_root', 'protocol', 'gate')))
    common.write_once(args.root / 'schedule.json', plan)
    print({'plan': str(args.root / 'schedule.json'), 'jobs': len(plan['jobs'])})


if __name__ == '__main__':
    main()
