import copy
from pathlib import Path
import tempfile
import unittest

from experiments import init_common as common
from experiments import init_extension_registry as registry


class RegistryTests(unittest.TestCase):
    def test_exact_task_grids_counts_sources_and_paired_controls(self):
        protocol=Path(registry.__file__).with_name('initialization_extension_protocol.md')
        total=0
        with tempfile.TemporaryDirectory() as folder:
            for task,expected_count in [('scifact',42),('nfcorpus',84),('aircraft',84),('extraction',42)]:
                grid=registry.build(task,protocol);rows=grid['candidates'];total+=len(rows)
                self.assertEqual(len(rows),expected_count)
                self.assertEqual(grid['parent_candidate_counts']['gme'],12*len(grid['learning_rates']))
                self.assertEqual(grid['rank'],2 if task=='aircraft' else 8)
                self.assertEqual(len({(r['family_id'],common.digest_json(r['spec']),r['learning_rate']) for r in rows}),len(rows))
                by_id={r['candidate_id']:r for r in rows}
                for row in rows:
                    control=by_id[row['comparison_control_id']]
                    self.assertEqual(control['learning_rate'],row['learning_rate'])
                    self.assertEqual(control['spec'],{'kind':'random','config':{'mode':'accepted'}})
                    self.assertEqual(row['initialization_lane'],'warmstart' if row['spec']['kind']=='extension_warmstart' else 'centered')
                path=Path(folder)/f'{task}.json';common.write(path,grid)
                self.assertEqual(registry.load_registry(path,protocol,task),grid)
                changed=copy.deepcopy(grid);changed['candidates'].pop();common.write(path,changed)
                with self.assertRaises(AssertionError):registry.load_registry(path,protocol,task)
        self.assertEqual(total,252)


if __name__=='__main__':unittest.main()
