"""Run disjoint Aircraft pilot shards, then select and finalize from all pilots.

This reuses the exact v2 model, data and training functions. Each shard writes
to its own directory; the finalizer checks all 80 pilot settings before it
touches the test split. The original serial runner need only finish indices 0:5.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import torch
from huggingface_hub import hf_hub_download

from experiments import pure_lora_variant_aircraft as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--shards', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    assert args.finalize != bool(args.indices)
    manifest = runner.read(args.root / 'manifest.json')
    for name, digest in manifest['source_sha256'].items():
        assert runner.sha(name) == digest, name
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache = Path('/var/tmp/dora-bench/cache/aircraft')
    args.epochs, args.pilot_epochs = 20, 10
    args.batch_size, args.head_lr, args.workers = 128, .001, 8
    args.reference_root = None
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.output != args.root:
        shutil.copy2(args.root / 'split_manifest.json', args.output / 'split_manifest.json')
    datasets, _ = runner.vision.datasets_and_manifest(args)
    assert runner.sha(args.output / 'split_manifest.json') == manifest['provenance']['split_manifest_sha256']
    checkpoint = hf_hub_download(runner.vision.REPO, 'model.safetensors',
        revision=runner.vision.REVISION, cache_dir='/var/tmp/dora-bench/cache/huggingface')
    assert runner.sha(checkpoint) == manifest['provenance']['checkpoint_sha256']
    provenance = dict(manifest['provenance'])
    assert provenance['torch'] == torch.__version__
    assert provenance['timm'] == __import__('timm').__version__
    assert provenance['torchvision'] == __import__('torchvision').__version__
    provenance['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    provenance['gpu'] = torch.cuda.get_device_name()
    provenance['source_sha256'] = manifest['source_sha256']
    receipt = {
        'parent_manifest_sha256': runner.sha(args.root / 'manifest.json'),
        'source_sha256': runner.sha(__file__),
        'cuda_visible_devices': provenance['cuda_visible_devices'],
        'indices': args.indices, 'finalize': args.finalize,
        'selection': 'validation only; unchanged 20 candidates and four LRs',
    }
    runner.write(args.output / ('finalizer.json' if args.finalize else 'shard.json'), receipt)
    rows = runner.candidates()
    if not args.finalize:
        for index in args.indices:
            candidate = rows[index]
            for lr in runner.vision.BASE_LRS:
                runner.vision.run_one(args, datasets, checkpoint, provenance,
                    candidate['id'], 2, 42, lr, 1., 'pilot',
                    backend=runner.backend(candidate, 42))
        runner.write(args.output / 'shard_complete.json', receipt)
        return

    pilots, paths = {}, []
    for candidate in rows:
        trials = []
        for lr in runner.vision.BASE_LRS:
            relative = Path('rank_2/pilots') / candidate['id'] / f'lr_{lr:g}_m_1/result.json'
            matches = [root / relative for root in [args.root, *args.shards]
                       if (root / relative).exists()]
            assert len(matches) == 1, (candidate['id'], lr, matches)
            path = matches[0]
            result = runner.read(path)
            assert result['method'] == candidate['id'] and result['seed'] == 42
            assert result['adapter_learning_rate'] == lr and result['epochs'] == 10
            assert result['frozen_base_unchanged'] and result['test'] is None
            assert not result['evaluate_test']
            assert runner.sha(path.parent / 'best_adapter_and_head.pt') == result['best_adapter_checkpoint_sha256']
            for diagnostic in result['adapter_diagnostics'].values():
                assert not diagnostic['dora_enabled']
                assert diagnostic['recipe_sha256'] == runner.recipe(candidate)['recipe_sha256']
            trials.append({'lr': lr, 'validation': result['best_validation'],
                           'result_path': str(path), 'result_sha256': runner.sha(path)})
            paths.append(str(path))
        winner = max(trials, key=lambda x: (x['validation']['macro_class_accuracy'],
                     -x['validation']['cross_entropy'], -x['lr']))
        pilots[candidate['id']] = {
            'candidate_id': candidate['id'], 'label': candidate['label'],
            'selected_lr': winner['lr'], 'selected_validation': winner['validation'],
            'trials': trials,
        }
    assert len(paths) == len(set(paths)) == 80
    runner.write(args.output / 'pilot_selection.json', pilots)
    winner_id = max(pilots, key=lambda key: (
        pilots[key]['selected_validation']['macro_class_accuracy'],
        -pilots[key]['selected_validation']['cross_entropy']))
    candidate = next(row for row in rows if row['id'] == winner_id)
    lr = pilots[winner_id]['selected_lr']
    runner.write(args.output / 'selection_lock.json', {
        'selected_id': winner_id, 'lr': lr,
        'pilot_selection_sha256': runner.sha(args.output / 'pilot_selection.json'),
        'test_used_for_selection': False,
    })
    finals = []
    for seed in manifest['seeds']:
        result = runner.vision.run_one(args, datasets, checkpoint, provenance,
            winner_id, 2, seed, lr, 1., 'final', backend=runner.backend(candidate, seed))
        finals.append({'candidate_id': winner_id, 'label': candidate['label'],
                       'seed': seed, 'lr': lr, 'result': result,
                       'recipe': runner.recipe(candidate, seed)})
        runner.write(args.output / 'final_results.json', finals)
    runner.write(args.output / 'completion.json', {
        'status': 'complete', 'selected_ids': [winner_id], 'pilot_count': len(pilots),
        'pilot_fit_count': len(paths), 'final_count': len(finals),
        'test_used_for_selection': False,
        'manifest_sha256': runner.sha(args.root / 'manifest.json'),
        'shards': [str(path) for path in args.shards],
    })


if __name__ == '__main__':
    main()
