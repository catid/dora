"""Compositional semantic parsing with validation-only, full-horizon tuning.

COGS is a public controlled-language task. Primary scoring compares complete
sets of logical atoms, preserving argument order, variable indices and definite
markers; strict whitespace-insensitive string match is also reported. This is
a subset experiment with a pretrained model, not the original COGS benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import time
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.extraction import batch, frozen_digest, nll, seed_all, strip_adapters, write_json
from experiments.second_round.generation_diagnostics import hit_generation_cap
from experiments.second_round_adapters import METHODS, inject_adapters, parameter_groups

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
MODEL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_REVISION = '165a7b669eade971fa47bf568a2e51925360fed8'
STRUCTURAL = {'cp_recursion', 'pp_recursion', 'obj_pp_to_subj_pp'}
SOURCES = ['experiments/second_round/cogs.py', 'experiments/second_round_adapters.py',
           'experiments/second_round/generation_diagnostics.py',
           'experiments/adapters.py', 'experiments/extraction.py', 'dora.py']
SYSTEM = ('Translate the sentence into the COGS logical-form notation. Output only the logical form. '
          'Use the exact predicate and role notation shown in the examples. Variables x _ i refer '
          'to the zero-based whitespace token position of the noun or verb in the input sentence. '
          'Mark definite nouns with *. Preserve entity names, argument order, and every relation. '
          'Use AND between conjuncts and semicolons after definite-noun declarations.')
ARGUMENT = r'(?:x\s*_\s*\d+|[A-Z][A-Za-z_]*)'
ATOM = re.compile(r'\s*\*?\s*[A-Za-z][A-Za-z_]*(?:\s*\.\s*[A-Za-z][A-Za-z_]*)*\s*\(\s*'
                  + ARGUMENT + r'(?:\s*,\s*' + ARGUMENT + r')*\s*\)\s*')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(text):
    return re.sub(r'\s+', '', text)


def atoms(text):
    """Reject extra prose/invalid syntax; allow reordering of flat conjuncts."""
    raw_parts = re.split(r'\bAND\b|;', text.strip())
    if not raw_parts or any(not ATOM.fullmatch(part) for part in raw_parts):
        return None
    parts = [compact(part) for part in raw_parts]
    if len(parts) != len(set(parts)):
        return None
    return frozenset(parts)


def prepare(args, tokenizer):
    data_dir = args.output / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    raw, hashes = {}, {}
    for split in ('train', 'dev', 'test', 'gen'):
        path = data_dir / (split + '.tsv')
        if not path.exists():
            url = f'https://raw.githubusercontent.com/najoungkim/COGS/{DATA_REVISION}/data/{split}.tsv'
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        hashes[split] = sha(path)
        raw[split] = []
        for index, line in enumerate(path.read_text().splitlines()):
            sentence, target, category = line.split('\t')
            raw[split].append({'id': f'{split}:{index}', 'utterance': sentence,
                               'target': target, 'category': category})
    # Preserve every scarce primitive/exposure example from the official train
    # split; filter evaluation overlap with the entire official training set.
    mandatory = [r for r in raw['train'] if r['category'] != 'in_distribution'] + raw['train'][:3]
    mandatory_ids = {r['id'] for r in mandatory}
    candidates = [r for r in raw['train'] if r['id'] not in mandatory_ids]
    selected = {'train': mandatory + random.Random(20261008).sample(candidates, args.train_size - len(mandatory))}
    demonstrations = raw['train'][:3]
    used = {compact(r['utterance']).lower() for r in raw['train']}
    removed = {}
    for split, limit in [('dev', args.val_size), ('test', args.test_size), ('gen', None)]:
        clean = []
        for row in raw[split]:
            key = compact(row['utterance']).lower()
            if key not in used:
                clean.append(row)
                used.add(key)
        removed[split] = len(raw[split]) - len(clean)
        if split == 'gen':
            categories = sorted({r['category'] for r in clean})
            selected[split] = []
            for category in categories:
                pool = [r for r in clean if r['category'] == category]
                selected[split].extend(random.Random(20261008).sample(pool, args.per_category))
        else:
            selected[split] = random.Random(20261008).sample(clean, limit)
    selected['validation'] = selected.pop('dev')
    encoded, lengths = {}, {}
    for split, rows in selected.items():
        encoded[split] = []
        lengths[split] = {'max_total': 0, 'max_target': 0}
        for row in rows:
            if row['category'] not in ('primitive',) and atoms(row['target']) is None:
                raise ValueError(f'Gold logical form rejected by metric: {row}')
            messages = [{'role': 'system', 'content': SYSTEM}]
            for example in demonstrations:
                messages.extend([{'role': 'user', 'content': example['utterance']},
                                 {'role': 'assistant', 'content': example['target']}])
            messages.append({'role': 'user', 'content': row['utterance']})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            continuation = tokenizer.encode(row['target'] + tokenizer.eos_token, add_special_tokens=False)
            lengths[split]['max_total'] = max(lengths[split]['max_total'], len(prefix) + len(continuation))
            lengths[split]['max_target'] = max(lengths[split]['max_target'], len(continuation))
            if len(prefix) + len(continuation) > args.max_length:
                raise ValueError(f'No silent truncation: {row["id"]} exceeds max_length')
            if split != 'train' and len(continuation) > args.max_new_tokens:
                raise ValueError(f'Generation cap below gold length: {row["id"]}')
            encoded[split].append({**row, 'prompt': prompt, 'input_ids': prefix + continuation,
                                   'labels': [-100] * len(prefix) + continuation})
    manifest = {k: [{field: r[field] for field in ('id', 'utterance', 'target', 'category')}
                    for r in rows] for k, rows in encoded.items()}
    path = args.output / 'split_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Split manifest differs; use a fresh directory')
    write_json(path, manifest)
    return encoded, {'repository': 'najoungkim/COGS', 'revision': DATA_REVISION,
                     'file_sha256': hashes, 'sizes': {k: len(v) for k, v in encoded.items()},
                     'evaluation_duplicates_removed': removed, 'token_lengths': lengths,
                     'all_155_primitive_and_exposure_rows_retained': True,
                     'demonstration_ids': [r['id'] for r in demonstrations],
                     'split_manifest_sha256': sha(path)}


@torch.inference_mode()
def evaluate(model, rows, tokenizer, args, path):
    model.eval()
    tokenizer.padding_side = 'left'
    eos_token_id = model.generation_config.eos_token_id
    # Sorting by prompt length only changes batching, not the chosen examples.
    ordered = sorted(rows, key=lambda r: len(r['input_ids']) - sum(x != -100 for x in r['labels']))
    predictions = []
    started = time.perf_counter()
    for start in range(0, len(ordered), args.generation_batch_size):
        part = ordered[start:start + args.generation_batch_size]
        inputs = tokenizer([r['prompt'] for r in part], padding=True, add_special_tokens=False,
                           return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id, use_cache=True)
        outputs = generated[:, inputs['input_ids'].shape[1]:]
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for row, text, output in zip(part, texts, outputs):
            token_ids = output.tolist()
            gold = atoms(row['target'])
            assert gold is not None
            prediction = atoms(text)
            guessed = prediction if prediction is not None else frozenset()
            predictions.append({'id': row['id'], 'category': row['category'], 'utterance': row['utterance'],
                                'gold': row['target'], 'text': text, 'valid': prediction is not None,
                                'atom_exact': prediction == gold, 'strict_exact': compact(text) == compact(row['target']),
                                'correct_atoms': len(guessed & gold), 'predicted_atoms': len(guessed),
                                'gold_atoms': len(gold),
                                'hit_generation_cap': hit_generation_cap(token_ids, args.max_new_tokens, eos_token_id),
                                'generation_diagnostics': {
                                    'version': 'eos_and_length_v1',
                                    'continuation_token_ids_with_padding': token_ids,
                                    'eos_token_id': eos_token_id,
                                    'pad_token_id': tokenizer.pad_token_id,
                                    'max_new_tokens': args.max_new_tokens}})
        print(json.dumps({'event': 'generation', 'file': str(path), 'done': min(start + len(part), len(ordered)), 'total': len(ordered)}), flush=True)
    torch.cuda.synchronize()
    path.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    categories = {}
    for category in sorted({r['category'] for r in predictions}):
        part = [r for r in predictions if r['category'] == category]
        categories[category] = {'count': len(part), 'atom_exact': statistics.mean(r['atom_exact'] for r in part),
                                'strict_exact': statistics.mean(r['strict_exact'] for r in part)}
    def group(names):
        values = [v['atom_exact'] for k, v in categories.items() if k in names]
        return statistics.mean(values) if values else None
    return {'count': len(predictions), 'atom_exact': statistics.mean(r['atom_exact'] for r in predictions),
            'strict_exact': statistics.mean(r['strict_exact'] for r in predictions),
            'valid': statistics.mean(r['valid'] for r in predictions),
            'atom_micro_f1': 2 * sum(r['correct_atoms'] for r in predictions) / max(1, sum(r['predicted_atoms'] + r['gold_atoms'] for r in predictions)),
            'category_macro_exact': statistics.mean(v['atom_exact'] for v in categories.values()),
            'structural_macro_exact': group(STRUCTURAL), 'lexical_macro_exact': group(set(categories) - STRUCTURAL),
            'categories': categories, 'generation_cap_hits': sum(r['hit_generation_cap'] for r in predictions),
            'seconds': time.perf_counter() - started, 'predictions_sha256': sha(path)}


def inject(model, method, seed, args):
    strip_adapters(model)
    seed_all(seed)
    inject_adapters(model, method, rank=args.rank, targets=('q_proj', 'v_proj'))


def restore(model, directory):
    state = torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[name])


def fit(model, method, seed, lr, multiplier, data, tokenizer, args, directory):
    directory.mkdir(parents=True, exist_ok=True)
    inject(model, method, seed, args)
    before = frozen_digest(model)
    groups = parameter_groups(model, lr, magnitude_lr_multiplier=multiplier, weight_decay=0.01)
    group_metadata = [{k: v for k, v in g.items() if k != 'params'} for g in groups]
    base_lrs = [g['lr'] for g in groups]
    optimizer = torch.optim.AdamW(groups)
    params = [p for p in model.parameters() if p.requires_grad]
    generator = torch.Generator().manual_seed(seed)
    order = []
    while len(order) < args.steps * args.batch_size:
        order.extend(torch.randperm(len(data['train']), generator=generator).tolist())
    best, best_step, best_state = math.inf, None, None
    model.train()
    warm = batch(data['train'][:args.batch_size], tokenizer)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(**warm, use_cache=False).loss.backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (directory / 'training.jsonl').open('w') as log:
        for step in range(1, args.steps + 1):
            model.train()
            inputs = batch([data['train'][i] for i in order[(step-1)*args.batch_size:step*args.batch_size]], tokenizer)
            ratio = min(1.0, step / max(1, args.steps * .1))
            for g, initial_lr in zip(optimizer.param_groups, base_lrs):
                g['lr'] = initial_lr * ratio
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**inputs, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gradient):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            record = {'step': step, 'loss': loss.item(), 'gradient_norm': float(gradient), 'lr': lr * ratio}
            if step % args.eval_every == 0 or step == args.steps:
                record['validation_nll'] = nll(model, data['validation'], tokenizer, args.batch_size)
                if record['validation_nll'] < best:
                    best, best_step = record['validation_nll'], step
                    best_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
            log.write(json.dumps(record) + '\n'); log.flush()
            if step == 1 or step % 32 == 0 or step == args.steps:
                print(json.dumps({'event': 'train', 'method': method, 'seed': seed, **record}), flush=True)
    torch.cuda.synchronize()
    elapsed, peak = time.perf_counter() - started, torch.cuda.max_memory_allocated()
    assert frozen_digest(model) == before, 'Frozen weights changed'
    torch.save(best_state, directory / 'adapter.pt')
    restore(model, directory)
    result = {'method': method, 'seed': seed, 'rank': args.rank, 'lr': lr, 'magnitude_lr_multiplier': multiplier,
              'steps': args.steps, 'selected_step': best_step, 'validation_nll': best,
              'train_and_validation_seconds': elapsed, 'peak_allocated_bytes': peak,
              'trainable_parameters': sum(p.numel() for p in params), 'optimizer_groups': group_metadata,
              'frozen_sha256': before, 'frozen_weights_unchanged': True, 'checkpoint_sha256': sha(directory / 'adapter.pt')}
    write_json(directory / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('/var/tmp/dora-bench/round2/cogs'))
    p.add_argument('--train-size', type=int, default=4096)
    p.add_argument('--val-size', type=int, default=128)
    p.add_argument('--test-size', type=int, default=128)
    p.add_argument('--per-category', type=int, default=32)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--steps', type=int, default=384)
    p.add_argument('--eval-every', type=int, default=96)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--generation-batch-size', type=int, default=32)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--max-new-tokens', type=int, default=640)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--learning-rates', type=float, nargs='+', default=[1e-4, 3e-4, 1e-3, 3e-3])
    p.add_argument('--mlr-learning-rates', type=float, nargs='+', default=[3e-4, 1e-3])
    p.add_argument('--magnitude-multipliers', type=float, nargs='+', default=[.01, .1])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--baseline-from', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    data, provenance = prepare(args, tokenizer)
    print(json.dumps({'event': 'dataset', **provenance}), flush=True)
    if args.prepare_only:
        return
    manifest = {'model': MODEL, 'model_revision': MODEL_REVISION, 'dataset': provenance,
                'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'source_sha256': {name: sha(name) for name in SOURCES},
                'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
                'protocol': 'Four full-horizon tuning trials per method at seed42; best validation target-token NLL selects rates and checkpoints. Winning seed42 checkpoint reused; seeds43/44 refit. IID and all21 OOD categories evaluated only after selection. No OOD data used for tuning. Primary atom-set exact match with strict-string exact match separately. BF16 frozen base, FP32 adapters; target q_proj/v_proj. Shared factor weight decay .01, no magnitude/gain decay.'}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Configuration/source mismatch; use a fresh output directory')
    write_json(manifest_path, manifest)
    for source in SOURCES:
        target = args.output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION,
                                                dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.requires_grad_(False)
    if args.smoke:
        result = fit(model, 'lora', args.seeds[0], 3e-4, 1., data, tokenizer, args, args.output / 'smoke')
        result['validation_generation'] = evaluate(model, data['validation'][:32], tokenizer, args, args.output / 'smoke_predictions.jsonl')
        write_json(args.output / 'smoke_result.json', result)
        print(json.dumps(result), flush=True)
        return
    baseline_file = args.output / 'baseline.json'
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
    elif args.baseline_from:
        baseline = json.loads(args.baseline_from.read_text())
        if (baseline['split_manifest_sha256'] != provenance['split_manifest_sha256']
                or baseline['model_revision'] != MODEL_REVISION
                or baseline['max_new_tokens'] != args.max_new_tokens):
            raise ValueError('Shared baseline protocol mismatch')
        write_json(baseline_file, baseline)
    else:
        baseline = {'method': 'baseline', 'iid': evaluate(model, data['test'], tokenizer, args, args.output / 'baseline_iid.jsonl'),
                    'ood': evaluate(model, data['gen'], tokenizer, args, args.output / 'baseline_ood.jsonl'),
                    'split_manifest_sha256': provenance['split_manifest_sha256'], 'model_revision': MODEL_REVISION,
                    'max_new_tokens': args.max_new_tokens}
        write_json(baseline_file, baseline)
    pilots, selected = [], {}
    for method in args.methods:
        candidates = ([(lr, mult) for lr in args.mlr_learning_rates for mult in args.magnitude_multipliers]
                      if method == 'dora_nora_mlr' else [(lr, 1.) for lr in args.learning_rates])
        assert len(candidates) == 4
        for index, (lr, mult) in enumerate(candidates):
            directory = args.output / f'pilot_{method}_{index}'
            result = (json.loads((directory / 'result.json').read_text()) if (directory / 'result.json').exists()
                      else fit(model, method, args.seeds[0], lr, mult, data, tokenizer, args, directory))
            pilots.append({**result, 'directory': str(directory)})
        selected[method] = min([r for r in pilots if r['method'] == method], key=lambda r: r['validation_nll'])
        write_json(args.output / 'tuning.json', {'trials': pilots, 'selected': selected})
    results = []
    for seed in args.seeds:
        for method in args.methods:
            winner = selected[method]
            directory = args.output / f'{method}_seed{seed}'
            complete = directory / 'final.json'
            if complete.exists():
                result = json.loads(complete.read_text())
            else:
                if seed == args.seeds[0]:
                    directory.mkdir(exist_ok=True)
                    shutil.copyfile(Path(winner['directory']) / 'adapter.pt', directory / 'adapter.pt')
                    inject(model, method, seed, args)
                    restore(model, directory)
                    result = {k: v for k, v in winner.items() if k != 'directory'}
                    result['reused_winning_full_budget_trial'] = winner['directory']
                else:
                    result = fit(model, method, seed, winner['lr'], winner['magnitude_lr_multiplier'], data, tokenizer, args, directory)
                result['iid'] = evaluate(model, data['test'], tokenizer, args, directory / 'iid_predictions.jsonl')
                result['ood'] = evaluate(model, data['gen'], tokenizer, args, directory / 'ood_predictions.jsonl')
                write_json(complete, result)
            results.append(result)
            write_json(args.output / 'results.json', {'manifest': manifest, 'baseline': baseline,
                                                     'tuning': pilots, 'selected': selected, 'results': results})
            print(json.dumps({'event': 'completed', 'method': method, 'seed': seed,
                              'iid_exact': result['iid']['atom_exact'], 'ood_macro': result['ood']['category_macro_exact'],
                              'structural': result['ood']['structural_macro_exact'], 'lexical': result['ood']['lexical_macro_exact']}), flush=True)


if __name__ == '__main__':
    main()
