"""Prevent quantized-base leakage between initializer extraction fits."""
import copy
import unittest

import torch
from torch import nn

from experiments import extraction as task
from experiments import init_common as common
from experiments import init_extraction as runner


class ExtractionIsolationTests(unittest.TestCase):
    def test_quantized_candidates_restore_identical_original_parameters(self):
        model=nn.Module();model.block=nn.Module()
        model.block.q_proj=nn.Linear(8,8,dtype=torch.bfloat16)
        model.block.v_proj=nn.Linear(8,4,dtype=torch.bfloat16)
        model.requires_grad_(False)
        original=runner.original_projection_parameters(model)
        self.assertEqual(len(original),2)
        original_values={name:weight.detach().clone() for name,(weight,bias) in original.items()}
        original_digest=task.frozen_digest(model)
        for method in ('quantizer_only','loftq','quantizer_only'):
            candidate={'candidate_id':method,'spec':{'kind':'quantized','config':{'method':method,'iterations':1}}}
            backend=common.backend_for(candidate,42)
            runner.restore_original_projections(model,backend,original,original_digest)
            torch.manual_seed(42)
            backend.inject_adapters(model,method,rank=2,targets=('q_proj','v_proj'))
            quantized_digest=task.frozen_digest(model)
            self.assertNotEqual(quantized_digest,original_digest)
            for name,module in model.named_modules():
                if name not in original:continue
                self.assertIsNot(module.weight,original[name][0])
                self.assertTrue(torch.equal(original[name][0],original_values[name]))
                with torch.no_grad():module.lora_A.add_(.1)
            runner.restore_original_projections(model,backend,original,original_digest)
            self.assertEqual(task.frozen_digest(model),original_digest)
            for name,module in model.named_modules():
                if name in original:self.assertIs(module.weight,original[name][0])

    def test_lane_validation_requires_separate_matched_quantizer_control(self):
        anchor={'candidate_id':'accepted','learning_rate':.0003,'spec':{'kind':'random','config':{'mode':'accepted'}}}
        control={'candidate_id':'q3','learning_rate':.0003,'spec':{'kind':'quantized','config':{
            'method':'quantizer_only','iterations':3,'base_source':'paired_loftq','magnitude_policy':'original'}}}
        loftq={'candidate_id':'loftq3','learning_rate':.0003,'comparison_control_id':'q3','spec':{'kind':'quantized','config':{
            'method':'loftq','iterations':3,'magnitude_policy':'original'}}}
        registry={'study_lane':'quantized','control_id':'accepted','candidates':[anchor,control,loftq]}
        self.assertEqual(runner.validate_study_lane(registry),'quantized')
        wrong=copy.deepcopy(registry);wrong['candidates'][1]['spec']['config']['base_source']='direct'
        with self.assertRaises(AssertionError):runner.validate_study_lane(wrong)
        wrong=copy.deepcopy(registry);wrong['candidates'][1]['spec']['config']['magnitude_policy']='quantized'
        with self.assertRaises(AssertionError):runner.validate_study_lane(wrong)
        wrong=copy.deepcopy(registry);wrong['study_lane']='centered'
        with self.assertRaises(AssertionError):runner.validate_study_lane(wrong)

    def test_accepted_random_cannot_be_replaced_in_warmstart_lane(self):
        registry={'study_lane':'warmstart','control_id':'accepted','candidates':[
            {'candidate_id':'accepted','learning_rate':.0003,'spec':{'kind':'random','config':{'mode':'accepted'}}},
            {'candidate_id':'one','spec':{'kind':'warmstart','config':{'method':'loraone'}}}]}
        self.assertEqual(runner.validate_study_lane(registry),'warmstart')
        registry['candidates'][0]['spec']['config']={'mode':'controlled'}
        with self.assertRaises(AssertionError):runner.validate_study_lane(registry)

    def test_shared_initializer_retains_two_learning_rate_control_pairs(self):
        shared={}
        comparisons=[{'candidate_id':'nf4_lr0001','initial_validation_nll_difference':-.1},
                     {'candidate_id':'nf4_lr0003','initial_validation_nll_difference':-.1}]
        for row in comparisons:runner.record_quantizer_comparison(shared,row['candidate_id'],row)
        self.assertEqual(shared['quantizer_controls'],{row['candidate_id']:row for row in comparisons})
        self.assertNotEqual(comparisons[0]['candidate_id'],comparisons[1]['candidate_id'])
        with self.assertRaises(AssertionError):
            runner.record_quantizer_comparison(shared,comparisons[0]['candidate_id'],{'changed':True})

    def test_numerical_failures_do_not_hide_infrastructure_or_shape_errors(self):
        for error in (RuntimeError('SVD failed to converge'),ValueError('Nonfinite factor storage')):
            self.assertTrue(runner.initialization_failure(error))
        for error in (RuntimeError('CUDA out of memory'),ValueError('Callback shape mismatch'),
                      RuntimeError('checkpoint checksum mismatch')):
            self.assertFalse(runner.initialization_failure(error))
            self.assertFalse(runner.numerical_failure(error))


if __name__=='__main__':unittest.main()
