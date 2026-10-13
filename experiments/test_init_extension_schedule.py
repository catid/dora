import tempfile
import unittest
from pathlib import Path

from experiments import init_common as common
from experiments import init_extension_registry as registry
from experiments import init_extension_schedule as scheduler


class ExtensionScheduleTests(unittest.TestCase):
    def test_full_grid_coverage_and_paired_control_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);protocol=root/'protocol';gate=root/'gate'
            protocol.write_text('protocol');gate.write_text('gate')
            for task in registry.TASKS:
                common.write_once(root/'registries'/f'{task}.json',registry.build(task,protocol))
            plan=scheduler.build_plan(root/'study',root/'source',root/'registries',protocol,gate,gate,root/'parents')
            jobs={job['job_id']:job for job in plan['jobs']}
            for task in registry.TASKS:
                prefix=f'extension_{task}_'
                pilots=[job for name,job in jobs.items() if name.startswith(prefix+'pilots_') and name!=prefix+'pilots_audit']
                actual=[]
                for job in pilots:
                    args=job['command'];actual+=args[args.index('--candidate-ids')+1:]
                    self.assertEqual(job['dependencies'],[prefix+'controls_audit'])
                expected=registry.build(task,protocol)['candidates']
                self.assertEqual(actual,[candidate['candidate_id'] for candidate in expected])
                self.assertEqual(jobs[prefix+'select']['dependencies'],[job['job_id'] for job in pilots])
                self.assertEqual(jobs[prefix+'finals']['dependencies'],[prefix+'pilots_audit'])
                self.assertNotIn('--family-ids',jobs[prefix+'finals']['command'])
                last=prefix+('finals' if task=='extraction' else 'finalize')
                self.assertEqual(jobs[prefix+'finals_audit']['dependencies'],[last])


if __name__=='__main__':unittest.main()
