"""Cost accounting tests with independently built raw receipts and bindings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments import init_cost_report as report


class CostReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        report.write(path, value)
        return path

    def envelope(self, tasks, candidates, sources, name='report', schema='dora-init-report-v1'):
        payload = {'schema': schema, 'fixture': True, 'tasks': tasks, 'all_candidates': candidates,
                   'evidence': {str(p): {'sha256': report.sha(p), 'bytes': p.stat().st_size} for p in sources}}
        path = self.save(f'{name}/report_inputs.json', payload)
        self.save(f'{name}/report_manifest.json', {'outputs': {'report_inputs.json': report.sha(path)}})
        return path

    def retrieval_fixture(self):
        root = self.root / 'run'
        directory = root / 'fits/a/seed_42'
        evaluation = {'nfcorpus': {'corpus_encoding_seconds': 3., 'total_seconds': 5.,
                                   'validation': {'query_encoding_and_ranking_seconds': 2.}},
                      'scifact': {'corpus_encoding_seconds': 4., 'total_seconds': 5.,
                                  'validation': {'query_encoding_and_ranking_seconds': 1.}}}
        pilot = {'candidate_id': 'a', 'seed': 42, 'status': 'pass', 'fit_id': 'fits/a/seed_42',
                 'training': {'seconds': 20.}, 'initialization': {'initializer_seconds': 1.5},
                 'initial_evaluation': {'nfcorpus': {'corpus_encoding_seconds': 1., 'total_seconds': 1.5,
                    'validation': {'query_encoding_and_ranking_seconds': .5}}},
                 'evaluation': evaluation, 'started_utc': '2026-01-01T00:00:00Z', 'finished_utc': '2026-01-01T00:00:30Z'}
        final = copy.deepcopy(pilot)
        final['finished_utc'] = '2026-01-02T00:00:00Z'
        for row in final['evaluation'].values():
            row['test'] = {'query_encoding_and_ranking_seconds': 7.}
            row['total_seconds'] += 7.
        first = self.save(directory / 'pilot.json', pilot)
        last = self.save(directory / 'result.json', final)
        normalized = {'candidate_id': 'a', 'seed': 42, 'status': 'pass', 'fit_id': str(directory)}
        task = {'training_task': 'nfcorpus', 'study_lane': 'centered', 'root': str(root), 'dataset': 'nfcorpus',
                'fits': {'a/seed_42': normalized}, 'cells': {
                    'family/fixed_original_lr': {'fit_keys': ['a/seed_42']},
                    'family/validation_tuned': {'fit_keys': ['a/seed_42']}}}
        transfer = copy.deepcopy(task); transfer['dataset'] = 'scifact_transfer'
        candidate = {'task': 'nfcorpus', 'study_lane': 'centered', 'candidate_id': 'a',
                     'family_id': 'family', 'pilot': {'status': 'pass', 'pilot_sha256': report.sha(first)},
                     'pilot_result_path': str(first), 'selected_lanes': ['fixed_original_lr', 'validation_tuned']}
        return {'centered/nfcorpus': task, 'centered/scifact_transfer': transfer}, [candidate], [first, last]

    def test_deduplicates_training_views_and_reports_but_retains_evaluation_invocations(self):
        tasks, candidates, sources = self.retrieval_fixture()
        path = self.envelope(tasks, candidates, sources)
        second = self.envelope(tasks, candidates, sources, name='second')
        output = report.collect([path, path, second])
        self.assertEqual(len(output['fits']), 1)
        self.assertEqual(len(output['inputs']), 2)
        fit = output['fits'][0]
        self.assertEqual(fit['training_seconds'], 20.)
        self.assertEqual(fit['initializer_seconds'], 1.5)
        self.assertTrue(fit['seed42_pilot_training_reused'])
        self.assertEqual(len(fit['snapshots']), 2)
        self.assertEqual(len(fit['evaluation_invocations']), 3)
        self.assertEqual({row['stage'] for row in fit['evaluation_invocations']}, {'initial', 'pilot', 'final'})
        final = next(row for row in fit['evaluation_invocations'] if row['stage'] == 'final')
        self.assertEqual(final['datasets']['nfcorpus']['reported_total_seconds'], 12.)
        self.assertEqual(final['datasets']['nfcorpus']['corpus_encoding_seconds'], 3.)
        self.assertEqual(final['datasets']['nfcorpus']['query_encoding_and_ranking_seconds'], {'validation': 2., 'test': 7.})
        self.assertNotIn('total_seconds', fit)
        self.assertEqual(len(next(row for row in fit['evaluation_invocations'] if row['stage'] == 'initial')['sources']), 2)

    def test_missing_and_zero_are_distinct_and_failed_fit_is_retained(self):
        root = self.root / 'run'
        failed = self.save('run/fits/a/seed_45/result.json', {'candidate_id': 'a', 'seed': 45,
                           'status': 'numerical_failure', 'fit_id': 'fits/a/seed_45',
                           'initialization': {'initializer_seconds': 0.}})
        normalized = {'candidate_id': 'a', 'seed': 45, 'status': 'numerical_failure', 'fit_id': str(failed.parent)}
        tasks = {'scifact': {'task': 'scifact', 'dataset': 'scifact', 'root': str(root), 'fits': {'a/seed_45': normalized}}}
        path = self.envelope(tasks, [], [failed], schema='dora-init-confirm-report-v1')
        payload = json.loads(path.read_text()); payload.pop('fixture'); payload['engineering_only'] = True
        report.write(path, payload)
        report.write(path.with_name('report_manifest.json'), {'outputs': {'report_inputs.json': report.sha(path)}})
        collected = report.collect([path]); self.assertTrue(collected['fixture'])
        fit = collected['fits'][0]
        self.assertEqual(fit['status'], 'numerical_failure')
        self.assertIsNone(fit['training_seconds'])
        self.assertEqual(fit['initializer_seconds'], 0.)
        self.assertEqual(fit['missing_cost_fields'], ['training_seconds'])
        self.assertEqual(fit['evaluation_invocations'], [])

    def test_report_hash_and_unused_evidence_hash_mismatch_are_rejected(self):
        tasks, candidates, sources = self.retrieval_fixture()
        unused = self.save('run/unused_audit.json', {'status': 'pass'})
        path = self.envelope(tasks, candidates, sources + [unused])
        unused.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            report.collect([path])
        path = self.envelope(tasks, candidates, sources)
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            report.collect([path])

    def test_raw_receipt_must_be_bound_and_repeated_training_must_agree(self):
        tasks, candidates, sources = self.retrieval_fixture()
        path = self.envelope(tasks, candidates, sources[:1])
        with self.assertRaisesRegex(ValueError, 'not bound'):
            report.collect([path])
        raw = json.loads(sources[1].read_text()); raw['training']['seconds'] = 999.
        report.write(sources[1], raw)
        path = self.envelope(tasks, candidates, sources)
        with self.assertRaisesRegex(ValueError, 'Conflicting repeated training_seconds'):
            report.collect([path])

    def test_extraction_receipt_links_are_verified_and_pilot_is_distinct_training(self):
        root = self.root / 'run'
        sources, candidates, fits = [], [], {}
        for stage, directory, seconds in [('pilot', 'pilots/a', 7.), ('final', 'finals/a/seed_42', 30.)]:
            result = self.save(f'run/{directory}/result.json', {'train_and_validation_seconds': seconds,
                               'test': {'generation_seconds': 11.} if stage == 'final' else None})
            initial = self.save(f'run/{directory}/initialization.json', {'initializer_seconds': .5,
                              'modules': {'q': {'elapsed_seconds': 800., 'factor_cache_hit': True}}})
            raw = {'candidate_id': 'a', 'seed': 42, 'status': 'pass', 'artifact_directory': directory,
                   'result_sha256': report.sha(result), 'initialization_sha256': report.sha(initial)}
            receipt = self.save(f'run/{directory}/receipt.json', raw); sources.append(receipt)
            if stage == 'final':
                fits['a/seed_42'] = {'candidate_id': 'a', 'seed': 42, 'status': 'pass', 'fit_id': str(receipt.parent)}
            else:
                candidates.append({'task': 'extraction', 'study_lane': 'centered', 'candidate_id': 'a',
                    'pilot': {'status': 'pass'}, 'pilot_result_path': str(receipt)})
        tasks = {'centered/extraction': {'training_task': 'extraction', 'study_lane': 'centered',
                 'root': str(root), 'dataset': 'extraction', 'fits': fits}}
        path = self.envelope(tasks, candidates, sources)
        output = report.collect([path]); self.assertEqual(len(output['fits']), 2)
        final = next(f for f in output['fits'] if f['stages'] == ['final'])
        self.assertEqual(final['training_seconds'], 30.)
        self.assertEqual(final['initializer_seconds'], .5)
        self.assertEqual(final['evaluation_invocations'][0]['seconds'], 11.)
        self.assertFalse(final['seed42_pilot_training_reused'])
        report.render(output, self.root / 'costs')
        self.assertTrue((self.root / 'costs/raw_costs.csv').is_file())
        (root / 'finals/a/seed_42/initialization.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            report.collect([path])

    def test_aircraft_training_scope_and_separate_test_timer(self):
        root = self.root / 'run'
        raw = {'candidate_id': 'a', 'seed': 42, 'status': 'pass', 'artifact_directory': 'fits/a/final/seed_42/artifacts',
               'result': {'training_wall_seconds': 50., 'test_seconds': 4.}, 'initialization': {}}
        path = self.save('run/fits/a/final/seed_42/fit.json', raw)
        tasks = {'extension/aircraft': {'training_task': 'aircraft', 'study_lane': 'extension', 'root': str(root),
                'fits': {'a/seed_42': {'candidate_id': 'a', 'seed': 42, 'status': 'pass', 'fit_id': str(path.parent)}}}}
        envelope = self.envelope(tasks, [], [path], schema='dora-init-extension-report-v1')
        fit = report.collect([envelope])['fits'][0]
        self.assertEqual(fit['training_seconds'], 50.)
        self.assertIn('validation', fit['training_scope'])
        self.assertEqual(fit['evaluation_invocations'][0]['seconds'], 4.)
        self.assertIsNone(fit['initializer_seconds'])

    def test_nonfinite_or_negative_duration_rejected(self):
        for value in (-1, float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                report.number(value, 'seconds')


if __name__ == '__main__':
    unittest.main()
