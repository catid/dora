import json
from pathlib import Path
import tempfile
import unittest
from experiments import audit_init_extension_metadata as checks


class AuditHistoryTests(unittest.TestCase):
    def test_exact_history_never_substitutes_rate_seed_or_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);a=root/'tuned';b=root/'fixed';a.mkdir();b.mkdir()
            (a/'result.json').write_text(json.dumps({'seed':42,'adapter_learning_rate':.0001,'epochs':20,'rank':2}))
            (b/'result.json').write_text(json.dumps({'seed':42,'adapter_learning_rate':.001,'epochs':20,'rank':2}))
            expected={'seed':42,'adapter_learning_rate':.001,'epochs':20,'rank':2}
            self.assertEqual(checks.exact_history([a,b],expected),b)
            self.assertIsNone(checks.exact_history([a,b],{**expected,'adapter_learning_rate':.0003}))
            self.assertIsNone(checks.exact_history([a,b],{**expected,'seed':43}))
            self.assertIsNone(checks.exact_history([a,b],{**expected,'epochs':10}))
            self.assertIsNone(checks.exact_history([root/'missing'],expected))

    def test_failure_audit_rejects_programming_and_resource_errors(self):
        for kind,message in [('RuntimeError','Nonfinite observed moments'),('ValueError','Nonfinite initializer factors'),('RuntimeError','linalg.eigh failed to converge')]:
            checks.numerical_failure({'type':kind,'message':message})
        for kind,message in [('RuntimeError','CUDA out of memory'),('ValueError','cache binding mismatch'),('KeyError','nonfinite'),('RuntimeError','Tensor size mismatch')]:
            with self.assertRaises(AssertionError):checks.numerical_failure({'type':kind,'message':message})


if __name__=='__main__':unittest.main()
