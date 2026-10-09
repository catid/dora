"""Regression cases for the two Qwen stop tokens and real length exhaustion."""

import unittest
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from experiments.second_round.generation_diagnostics import hit_generation_cap


class GenerationDiagnosticsTests(unittest.TestCase):
    def test_both_qwen_stop_tokens_suppress_cap_flag(self):
        # The prior tokenizer-only check missed 151643 in an unpadded or
        # longest-in-batch response. Either configured stop ID must suffice.
        for eos in (151645, 151643):
            with self.subTest(eos=eos):
                self.assertFalse(hit_generation_cap([42, eos], 2, [151645, 151643]))
                self.assertFalse(hit_generation_cap([42, eos, 0, 0], 4, [151645, 151643]))

    def test_eos_at_last_allowed_token_is_not_length_exhaustion(self):
        for config in (151643, [151645, 151643]):
            self.assertFalse(hit_generation_cap([10, 11, 151643], 3, config))

    def test_below_limit_without_eos_is_not_a_cap_hit(self):
        for config in (151645, [151645, 151643], None, []):
            self.assertFalse(hit_generation_cap([10, 11], 3, config))
            self.assertFalse(hit_generation_cap([], 3, config))

    def test_true_cap_without_any_configured_eos(self):
        for config in (151645, [151645, 151643], None, []):
            self.assertTrue(hit_generation_cap([10, 11, 12], 3, config))
        # An ID only counts as a stop when it was configured as one.
        self.assertTrue(hit_generation_cap([10, 151643], 2, 151645))
        self.assertTrue(hit_generation_cap([10, 151643], 2, None))

    def test_invalid_budget_or_eos_configuration_is_rejected(self):
        for value in (0, -1, 3.5, True):
            with self.assertRaises(ValueError):
                hit_generation_cap([10], value, 151645)
        for value in (True, "151645", [151645, "151643"]):
            with self.assertRaises(TypeError):
                hit_generation_cap([10], 1, value)

    def test_evaluator_preserves_decoding_and_records_auditable_stop_metadata(self):
        import torch
        from experiments.second_round.cogs import evaluate

        continuation = torch.tensor([[41, 42, 151643], [41, 151643, 151645],
                                     [41, 42, 43], [41, 42, 151645]])
        calls = []
        case = self

        class Inputs(dict):
            def to(self, device):
                case.assertEqual(device, 'cuda')
                return self

        class Tokenizer:
            pad_token_id = eos_token_id = 151645

            def __call__(self, prompts, **kwargs):
                case.assertEqual(kwargs, {'padding': True, 'add_special_tokens': False,
                                          'return_tensors': 'pt'})
                return Inputs(input_ids=torch.ones((len(prompts), 2), dtype=torch.long))

            def batch_decode(self, outputs, **kwargs):
                case.assertTrue(torch.equal(outputs, continuation))
                case.assertEqual(kwargs, {'skip_special_tokens': True})
                return ['dog(x_1)'] * len(outputs)

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

            def eval(self):
                return self

            def generate(self, input_ids, **kwargs):
                calls.append(kwargs)
                return torch.cat((input_ids, continuation), dim=1)

        rows = [{'id': str(i), 'category': 'lexical', 'utterance': 'dog',
                 'target': 'dog(x_1)', 'prompt': 'prompt', 'input_ids': [1, 2],
                 'labels': [-100, 2]} for i in range(4)]
        args = SimpleNamespace(generation_batch_size=4, max_new_tokens=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            with patch('torch.autocast', return_value=nullcontext()), \
                    patch('torch.cuda.synchronize'), redirect_stdout(io.StringIO()):
                result = evaluate(Model(), rows, Tokenizer(), args, path)
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(calls, [{'do_sample': False, 'max_new_tokens': 3,
                                 'pad_token_id': 151645, 'use_cache': True}])
        self.assertEqual(result['generation_cap_hits'], 1)
        self.assertEqual(result['atom_exact'], 1)
        self.assertEqual(result['strict_exact'], 1)
        self.assertEqual([row['hit_generation_cap'] for row in predictions],
                         [False, False, True, False])
        for row, token_ids in zip(predictions, continuation.tolist()):
            self.assertEqual(row['generation_diagnostics'], {
                'version': 'eos_and_length_v1',
                'continuation_token_ids_with_padding': token_ids,
                'eos_token_id': [151645, 151643], 'pad_token_id': 151645,
                'max_new_tokens': 3})


if __name__ == "__main__":
    unittest.main()
