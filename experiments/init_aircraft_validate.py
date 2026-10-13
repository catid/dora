"""Real-data ViT parity between accepted DoRA and the initializer control."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import traceback

import torch
from torch.nn import functional as F
import timm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from experiments import init_ablations as initializers
from experiments import init_common as common
from experiments import nora_current as accepted
from experiments.second_round import vision
from experiments.vision import seed_everything, tensor_digest


def compare(left, right, *, gradients=False):
    first, second = dict(left.named_parameters()), dict(right.named_parameters())
    assert list(first) == list(second), 'Parameter registration order changed'
    count = 0
    for name, value in first.items():
        other = second[name]
        assert value.requires_grad == other.requires_grad, name
        assert value.dtype == other.dtype and value.shape == other.shape, name
        if gradients:
            value, other = value.grad, other.grad
            assert (value is None) == (other is None), name
        if value is not None:
            assert torch.equal(value, other), (name, float((value-other).abs().max()))
            count += 1
    return {'tensors_compared': count, 'max_absolute_error': 0., 'bitwise_equal': True}


def optimizer(model, backend):
    groups = backend.parameter_groups(model, .0005)
    groups.append({'params': list(model.head.parameters()), 'param_names': ['head.weight', 'head.bias'],
                   'lr': .0005, 'weight_decay': .01})
    return torch.optim.AdamW(groups)


def backward(model, opt, images, labels):
    model.train()
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(images)
        loss = F.cross_entropy(logits.float(), labels, label_smoothing=.1)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    assert torch.isfinite(norm)
    return logits.detach(), loss.detach(), norm.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate-evidence', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    parser.add_argument('--cache', type=Path, default=Path('/var/tmp/dora-bench/cache/aircraft'))
    args = parser.parse_args()
    gate = common.read(args.gate_evidence)
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    for name, digest in gate['source_sha256'].items():
        assert common.sha(name) == digest, f'Backend gate source changed: {name}'
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'task_parity.json').exists(), 'Use a new output for a fresh validation'
    source_names = list(dict.fromkeys([*gate['source_sha256'], 'experiments/init_aircraft_validate.py',
                        'experiments/second_round/vision.py', 'experiments/vision.py',
                        'experiments/second_round_adapters.py', 'experiments/original_dora.py']))
    source_hashes = {name: common.sha(name) for name in source_names}
    for name in source_names:
        destination = args.output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert common.sha(destination) == source_hashes[name]
        else:
            shutil.copyfile(name, destination)
    shutil.copyfile(args.gate_evidence, args.output/'backend_gate.json')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    source = args.reference_root/'split_manifest.json'
    destination = args.output/'split_manifest.json'
    if destination.exists():
        assert common.sha(destination) == common.sha(source)
    else:
        shutil.copyfile(source, destination)
    datasets, split_manifest = vision.datasets_and_manifest(args)
    checkpoint = hf_hub_download(vision.REPO, 'model.safetensors', revision=vision.REVISION,
                                cache_dir='/var/tmp/dora-bench/cache/huggingface')
    seed_everything(42)
    base = timm.create_model(vision.MODEL, pretrained=False)
    base.load_state_dict(load_file(checkpoint), strict=True)
    base.reset_classifier(100)
    reference_path = args.reference_root/'baseline/seed_42/result.json'
    reference = common.read(reference_path)
    assert tensor_digest(base.head.state_dict().items()) == reference['initial_head_sha256']
    assert tensor_digest((n,p) for n,p in base.named_parameters() if not n.startswith('head.')) == reference['initial_backbone_sha256']
    initial_rng = torch.get_rng_state()
    batches, input_records = [], []
    for index in range(3):
        seed_everything(20261009 + index)
        samples = [datasets['train'][i] for i in range(4)]
        images = torch.stack([sample[0] for sample in samples])
        labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
        input_records.append({'augmentation_seed': 20261009+index, 'train_indices': list(range(4)),
                              'labels': labels.tolist(), 'tensor_sha256': tensor_digest([('images',images),('labels',labels)])})
        batches.append((images.cuda(), labels.cuda()))
    backend = initializers.backend(initializers.InitSpec(), factor_seed=42)
    case = {'name':'accepted_default', 'reference':'current_dora', 'candidate':'accepted_init1',
            'reference_backend':'nora_current', 'candidate_recipe':backend.recipe_manifest('accepted_init1'), 'steps':[]}
    try:
        left, right = copy.deepcopy(base), copy.deepcopy(base)
        torch.set_rng_state(initial_rng)
        accepted.inject_adapters(left, 'current_dora', rank=2,
                                 targets=lambda name,module: name.endswith(vision.TARGET_SUFFIXES))
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(initial_rng)
        backend.inject_adapters(right, 'accepted_init1', rank=2,
                                targets=lambda name,module: name.endswith(vision.TARGET_SUFFIXES))
        assert torch.equal(expected_rng, torch.get_rng_state()), 'Initialization RNG consumption changed'
        left.head.requires_grad_(True); right.head.requires_grad_(True)
        case['initial_parameters'] = compare(left, right)
        assert all(p.dtype == torch.float32 for model in (left,right) for p in model.parameters())
        left.cuda(); right.cuda()
        frozen_before = tensor_digest((name,p) for name,p in left.named_parameters() if not p.requires_grad)
        left_opt, right_opt = optimizer(left, accepted), optimizer(right, backend)
        for index, (images,labels) in enumerate(batches):
            cuda_rng = torch.cuda.get_rng_state()
            left_values = backward(left, left_opt, images, labels)
            after_rng = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(cuda_rng)
            right_values = backward(right, right_opt, images, labels)
            assert torch.equal(after_rng, torch.cuda.get_rng_state()), 'Training RNG consumption changed'
            for first,second in zip(left_values,right_values):
                assert torch.equal(first,second), float((first-second).abs().max())
            gradients = compare(left, right, gradients=True)
            left_opt.step(); right_opt.step()
            updates = compare(left, right)
            case['steps'].append({'step':index+1, 'loss':float(left_values[1]),
                                  'preclip_gradient_norm':float(left_values[2]),
                                  'logits_loss_clip_bitwise_equal':True, 'gradients':gradients,
                                  'parameters_after_adamw':updates})
        assert frozen_before == tensor_digest((name,p) for name,p in right.named_parameters() if not p.requires_grad)
        case['candidate_norm_diagnostics'] = backend.assert_no_clamp(right)
        case['frozen_parameters_unchanged'] = True
        case['passed'] = True
    except Exception as error:
        case.update(passed=False, error=str(error), traceback=traceback.format_exc())
    assert source_hashes == {name:common.sha(name) for name in source_names}, 'Sources changed during validation'
    payload = {'schema':'dora-init-aircraft-task-parity-v1', 'passed':case['passed'],
               'device':torch.cuda.get_device_name(0), 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
               'torch':torch.__version__, 'autocast':'bfloat16', 'base_and_adapter_dtype':'float32',
               'model':vision.MODEL, 'model_revision':vision.REVISION, 'checkpoint_sha256':common.sha(checkpoint),
               'initial_head_sha256':reference['initial_head_sha256'], 'initial_backbone_sha256':reference['initial_backbone_sha256'],
               'reference_result_sha256':common.sha(reference_path), 'split_manifest_sha256':common.sha(destination),
               'backend_gate_sha256':common.sha(args.gate_evidence), 'source_sha256':source_hashes,
               'batch_size':4, 'rank':2, 'seed':42,
               'optimizer':'AdamW; factor/head decay .01, magnitude decay0; first-epoch warmup LR .0005',
               'label_smoothing':.1, 'gradient_clip':1., 'inputs':input_records, 'cases':[case],
               'scope':'Three matched real Aircraft train minibatches, full ViT backward/clip/AdamW. Binds accepted initializer control to historical accepted DoRA with bitwise initial parameters, RNG, logits, loss, gradients and updates. Other initializer formula and callback checks are in the separately bound backend gate. This is a parity check, not a downstream accuracy result.'}
    common.write_once(args.output/'task_parity.json', payload)
    print(json.dumps({'case':case['name'], 'passed':case['passed'], 'error':case.get('error')}), flush=True)
    raise SystemExit(0 if payload['passed'] else 1)


if __name__ == '__main__':
    main()
