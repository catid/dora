"""Real-ViT task controls for endpoint equivalence and exact FP32 no-op reuse."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import traceback

import torch
from torch.nn import functional as F
import timm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from experiments import change_ablations as changes
from experiments import second_round_adapters as historical
from experiments.second_round import vision
from experiments.vision import file_sha256, seed_everything, tensor_digest, write_json


def canonical_parameters(model, original_names=False):
    result = {}
    for name, parameter in model.named_parameters():
        if original_names and name.endswith('.lora_A'):
            name = name[:-6] + 'lora_B'
        elif original_names and name.endswith('.lora_B'):
            name = name[:-6] + 'lora_A'
        result[name] = parameter
    return result


def compare(a, b, original_names=False, gradients=False):
    left, right = canonical_parameters(a, original_names), canonical_parameters(b)
    assert list(left) == list(right), 'Parameter registration order changed'
    count, max_error = 0, 0.0
    for name, first in left.items():
        second = right[name]
        assert first.requires_grad == second.requires_grad
        if not first.requires_grad:
            continue
        first, second = (first.grad, second.grad) if gradients else (first, second)
        assert (first is None) == (second is None), name
        if first is not None:
            assert first.dtype == second.dtype and first.shape == second.shape
            difference = (first.detach() - second.detach()).abs().max().item()
            max_error = max(max_error, difference)
            assert torch.equal(first, second), (name, difference)
            count += 1
    return {'tensors_compared': count, 'max_absolute_error': max_error, 'bitwise_equal': True}


def optimizer(model, module):
    groups = module.parameter_groups(model, .0005)
    groups.append({'params': list(model.head.parameters()), 'param_names': ['head.weight', 'head.bias'],
                   'lr': .0005, 'weight_decay': .01})
    return torch.optim.AdamW(groups)


def step(model, opt, images, labels):
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
    parser.add_argument('--reference-root', type=Path, default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    parser.add_argument('--cache', type=Path, default=Path('/var/tmp/dora-bench/cache/aircraft'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    source = args.reference_root / 'split_manifest.json'
    (args.output / 'split_manifest.json').write_bytes(source.read_bytes())
    datasets, manifest = vision.datasets_and_manifest(args)
    checkpoint = hf_hub_download(vision.REPO, 'model.safetensors', revision=vision.REVISION,
                                cache_dir='/var/tmp/dora-bench/cache/huggingface')
    seed_everything(42)
    base = timm.create_model(vision.MODEL, pretrained=False)
    base.load_state_dict(load_file(checkpoint), strict=True)
    base.reset_classifier(100)
    reference = json.loads((args.reference_root / 'baseline/seed_42/result.json').read_text())
    assert tensor_digest(base.head.state_dict().items()) == reference['initial_head_sha256']
    assert tensor_digest((n,p) for n,p in base.named_parameters() if not n.startswith('head.')) == reference['initial_backbone_sha256']
    init_rng = torch.get_rng_state()
    batches, input_records = [], []
    for index in range(3):
        seed_everything(20261009 + index)
        samples = [datasets['train'][i] for i in range(4)]
        images = torch.stack([sample[0] for sample in samples])
        labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
        input_records.append({'augmentation_seed': 20261009 + index, 'train_indices': list(range(4)),
                              'labels': labels.tolist(), 'tensor_sha256': tensor_digest([('images', images), ('labels', labels)])})
        batches.append((images.cuda(), labels.cuda()))
    cases = [('original_endpoint', 'dora_original', 'ab_original', True),
             ('revised_endpoint', 'dora', 'ab_revised', True),
             ('original_magnitude_fp32_noop', 'ab_original', 'ab_m_fp32', False),
             ('cumulative_magnitude_fp32_noop', 'cum_axis_detach', 'cum_axis_detach_m_fp32', False)]
    results = []
    for name, left_method, right_method, reference_backend in cases:
        record = {'name': name, 'reference': left_method, 'candidate': right_method,
                  'reference_backend': 'second_round_adapters' if reference_backend else 'change_ablations',
                  'steps': []}
        try:
            left, right = copy.deepcopy(base), copy.deepcopy(base)
            torch.set_rng_state(init_rng)
            (historical if reference_backend else changes).inject_adapters(left, left_method, rank=2,
                targets=lambda n,m: n.endswith(vision.TARGET_SUFFIXES))
            left_rng = torch.get_rng_state()
            torch.set_rng_state(init_rng)
            changes.inject_adapters(right, right_method, rank=2,
                targets=lambda n,m: n.endswith(vision.TARGET_SUFFIXES))
            assert torch.equal(left_rng, torch.get_rng_state()), 'Initialization RNG consumption changed'
            left.head.requires_grad_(True); right.head.requires_grad_(True)
            original_names = left_method == 'dora_original'
            record['initial_parameters'] = compare(left, right, original_names)
            assert all(p.dtype == torch.float32 for m in (left,right) for p in m.parameters())
            left.cuda(); right.cuda()
            left_opt = optimizer(left, historical if reference_backend else changes)
            right_opt = optimizer(right, changes)
            for index, (images, labels) in enumerate(batches):
                state = torch.cuda.get_rng_state()
                left_values = step(left, left_opt, images, labels)
                left_after_rng = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(state)
                right_values = step(right, right_opt, images, labels)
                assert torch.equal(left_after_rng, torch.cuda.get_rng_state())
                for a,b in zip(left_values,right_values):
                    assert torch.equal(a,b), float((a-b).abs().max())
                gradients = compare(left,right,original_names,gradients=True)
                left_opt.step(); right_opt.step()
                updates = compare(left,right,original_names)
                record['steps'].append({'step': index+1, 'loss': float(left_values[1]),
                                        'preclip_gradient_norm': float(left_values[2]),
                                        'logits_loss_clip_bitwise_equal': True,
                                        'gradients': gradients, 'parameters_after_adamw': updates})
            record['candidate_norm_diagnostics'] = changes.assert_no_clamp(right)
            record['passed'] = True
        except Exception as error:
            record['passed'] = False
            record['error'] = str(error)
            record['traceback'] = traceback.format_exc()
        results.append(record)
        print(json.dumps({'case':name,'passed':record['passed'],'error':record.get('error')}),flush=True)
        del left, right
        if 'left_opt' in locals():
            del left_opt, right_opt
        torch.cuda.empty_cache()
    payload = {'passed': all(row['passed'] for row in results), 'device': torch.cuda.get_device_name(0),
               'torch': torch.__version__, 'autocast': 'bfloat16', 'base_and_adapter_dtype': 'float32',
               'model': vision.MODEL, 'checkpoint_sha256': file_sha256(checkpoint),
               'initial_head_sha256': reference['initial_head_sha256'], 'initial_backbone_sha256': reference['initial_backbone_sha256'],
               'batch_size': 4, 'rank': 2, 'seed': 42, 'optimizer': 'AdamW; factor/head decay .01, magnitude decay0; first-epoch warmup LR .0005',
               'label_smoothing': .1, 'gradient_clip': 1., 'inputs': input_records, 'cases': results,
               'source_sha256': {name:file_sha256(name) for name in ('experiments/change_aircraft_validate.py','experiments/change_ablations.py','experiments/second_round_adapters.py','experiments/original_dora.py','experiments/adapters.py','dora_row.py')},
               'scope': 'Three matched real-data minibatches from the official Aircraft train split, full ViT backward/clip/AdamW; bitwise checks support endpoint and exact FP32 magnitude-storage no-op controls. These checks do not measure downstream accuracy.'}
    write_json(args.output/'task_parity.json',payload)
    (args.output/'task_parity_source.py').write_bytes(Path(__file__).read_bytes())
    raise SystemExit(0 if payload['passed'] else 1)


if __name__ == '__main__':
    main()
