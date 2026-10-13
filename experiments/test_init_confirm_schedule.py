"""Verify confirmation dependencies and exclusive full-budget seed shards."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import init_confirm_schedule as scheduler


class ConfirmationScheduleTests(unittest.TestCase):
    def test_task_dags_gate_every_fit_and_audit_complete_seed_set(self):
        for task in ('scifact', 'nfcorpus', 'aircraft', 'extraction'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                protocol, gate = root/'protocol', root/'gate'
                protocol.write_text('protocol'); gate.write_text('gate')
                registry = {'task': task, 'seeds': [45, 46, 47, 48, 49]}
                with mock.patch.object(scheduler.confirmation, 'derive', return_value=registry) as derive:
                    plan = scheduler.build_plan(root/'schedule', root/'source', root/'parent', protocol, gate)
                derive.assert_called_once_with(root/'parent', protocol)
                jobs = {job['job_id']: job for job in plan['jobs']}
                prefix = f'confirm_{task}_'
                fits = [job for name, job in jobs.items() if name.startswith(prefix+'seed_')]
                self.assertEqual(len(fits), 5)
                for seed, job in zip(registry['seeds'], fits):
                    self.assertEqual(job['dependencies'], [prefix+'controls_audit'])
                    command = job['command']
                    self.assertEqual(command[command.index('--seeds')+1:], [str(seed)])
                    self.assertEqual(command[command.index('--phase')+1], 'finals' if task == 'extraction' else 'fits')
                    self.assertEqual(job['source'], str(root/'source'))
                self.assertEqual(jobs[prefix+'controls_audit']['dependencies'], [prefix+'controls'])
                last = jobs[prefix+'finals_audit']
                if task == 'extraction':
                    self.assertEqual(set(last['dependencies']), {job['job_id'] for job in fits})
                    self.assertTrue(last['gpu'])
                else:
                    self.assertEqual(last['dependencies'], [prefix+'finalize'])
                    self.assertEqual(set(jobs[prefix+'finalize']['dependencies']), {job['job_id'] for job in fits})
                    self.assertFalse(last['gpu'])

    def test_unready_parent_does_not_create_a_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(scheduler.confirmation, 'derive', side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError):
                    scheduler.build_plan(root/'schedule', root, root, root, root)
            self.assertFalse((root/'schedule').exists())


if __name__ == '__main__':
    unittest.main()
