"""Semantic-scoring safeguards for the COGS subset experiment."""
import unittest

from experiments.second_round.cogs import atoms, compact


class CogsScoringTests(unittest.TestCase):
    def test_conjunction_order_and_whitespace_do_not_change_atoms(self):
        a = '* dog ( x _ 1 ) ; run . agent ( x _ 2 , x _ 1 )'
        b = 'run.agent(x_2,x_1) AND *dog(x_1)'
        self.assertEqual(atoms(a), atoms(b))
        self.assertNotEqual(compact(a), compact(b))

    def test_roles_arguments_definiteness_and_indices_remain_distinct(self):
        gold = atoms('*dog(x_1); help.agent(x_2,x_1) AND help.theme(x_2,Emma)')
        for wrong in ('*dog(x_1); help.agent(x_1,x_2) AND help.theme(x_2,Emma)',
                      'dog(x_1); help.agent(x_2,x_1) AND help.theme(x_2,Emma)',
                      '*dog(x_1); help.theme(x_2,x_1) AND help.agent(x_2,Emma)',
                      '*dog(x_3); help.agent(x_2,x_1) AND help.theme(x_2,Emma)'):
            self.assertNotEqual(gold, atoms(wrong))

    def test_invalid_prose_duplicates_and_partial_forms_are_rejected(self):
        for wrong in ('', 'The answer is dog(x_1)', 'dog(x_1) AND dog(x_1)',
                      'dog(x_1) AND', 'help.agent(x_2,)', 'dog(x_1); garbage'):
            self.assertIsNone(atoms(wrong), wrong)


if __name__ == '__main__':
    unittest.main()
