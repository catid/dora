"""Schedule the separately registered LoRA-Null/GME extension."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from experiments import init_common as common
from experiments import init_extension_registry as registry_builder
from experiments.init_schedule import chunks


def build_plan(root, source, registry_root, protocol, gate, aircraft_gate,
               parent_root, tasks=('scifact', 'nfcorpus', 'extraction', 'aircraft')):
    jobs = []
    for task in tasks:
        registry_path = registry_root/f'{task}.json'
        registry = registry_builder.load_registry(registry_path, protocol, task)
        output = root/'runs'/task
        module = 'retrieval' if task in ('scifact', 'nfcorpus') else task
        base = [sys.executable, '-m', f'experiments.init_extension_{module}',
                '--registry', str(registry_path), '--protocol', str(protocol),
                '--output', str(output), '--gate-evidence', str(gate)]
        base += ['--calibration-reference' if task == 'extraction' else '--study-root',
                 str(parent_root/task)]
        if module == 'retrieval':
            base += ['--task', task]
        if task == 'aircraft':
            base += ['--task-gate-evidence', str(aircraft_gate)]
        audit = [sys.executable, '-m', f'experiments.audit_init_extension_{module}',
                 '--root', str(output)]

        def add(suffix, command, dependencies, gpu=True, priority=1):
            jid = f'extension_{task}_{suffix}'
            jobs.append(dict(job_id=jid, command=command, dependencies=dependencies,
                             gpu=gpu, priority=priority, task=task, study_lane='extension',
                             source=str(source), output=str(output),
                             registry_sha256=common.sha(registry_path)))
            return jid

        controls = add('controls', base+['--phase', 'controls'], [])
        control_audit = add('controls_audit', audit+['--phase', 'controls'],
                            [controls], gpu=task == 'extraction')
        pilots = []
        for index, candidates in enumerate(chunks([c['candidate_id'] for c in registry['candidates']],
                                                  32 if task == 'aircraft' else 48)):
            pilots.append(add(f'pilots_{index:03}', base+['--phase', 'pilots', '--candidate-ids', *candidates],
                              [control_audit], priority=2))
        selection = add('select', base+['--phase', 'select'], pilots, gpu=task == 'extraction')
        pilot_audit = add('pilots_audit', audit+['--phase', 'pilots'], [selection], gpu=task == 'extraction')
        # Every family may request the same-LR anchor. Keep all families for a
        # task in one finals shard, avoiding shared comparison checkpoint races.
        final = add('finals', base+['--phase', 'finals'], [pilot_audit], priority=2)
        if task != 'extraction':
            final = add('finalize', base+['--phase', 'finalize'], [final], gpu=False)
        add('finals_audit', audit+['--phase', 'finals'], [final], gpu=task == 'extraction')
    return dict(schema='dora-init-schedule-v1', source=str(source), root=str(root),
                protocol_sha256=common.sha(protocol), gate_sha256=common.sha(gate),
                aircraft_gate_sha256=common.sha(aircraft_gate), parent_root=str(parent_root), jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'registry-root', 'protocol', 'gate', 'aircraft-gate', 'parent-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['scifact', 'nfcorpus', 'extraction', 'aircraft'])
    args = parser.parse_args()
    plan = build_plan(*(getattr(args, name).resolve() for name in
                        ('root', 'source', 'registry_root', 'protocol', 'gate', 'aircraft_gate', 'parent_root')),
                      tasks=args.tasks)
    common.write_once(args.root/'schedule.json', plan)
    print({'plan': str(args.root/'schedule.json'), 'jobs': len(plan['jobs'])})


if __name__ == '__main__':
    main()
