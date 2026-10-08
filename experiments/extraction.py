"""Compare four adapters on held-out ViGGO utterance-to-JSON extraction.

CUDA_VISIBLE_DEVICES=2 HF_HOME=/var/tmp/dora-bench/cache/huggingface \
  python -m experiments.extraction --output /var/tmp/dora-bench/runs/extraction

The public ViGGO data-to-text pairs are reversed into an extraction task.
Official splits are retained; identical utterances are removed across splits.
Learning rates are selected by validation target-token NLL, never test scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
from experiments.adapters import AdapterLinear, METHODS, inject_adapters

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
MODEL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATASET = 'GEM/viggo'
DATASET_REVISION = 'c851cd5ff2ee92f0137fcf24014e37427a2d30b7'
ALLOWED_KEYS = {'dialogue_act', 'name', 'release_year', 'developer', 'esrb', 'rating',
                'genres', 'player_perspective', 'has_multiplayer', 'platforms',
                'available_on_steam', 'has_linux_release', 'has_mac_release', 'exp_release_date', 'specifier'}
ALLOWED_ACTS = {'inform', 'confirm', 'recommend', 'request', 'give_opinion',
                'verify_attribute', 'suggest', 'request_explanation', 'request_attribute'}
SYSTEM = (
    'Extract the dialogue act and video-game attributes from the utterance. '
    'Return only one JSON object. Use key "dialogue_act" with one of inform, '
    'confirm, recommend, request, give_opinion, verify_attribute, suggest, '
    'request_explanation, request_attribute. Other allowed keys are name, '
    'release_year, developer, esrb, rating, genres, player_perspective, '
    'has_multiplayer, platforms, available_on_steam, has_linux_release, '
    'has_mac_release, exp_release_date, specifier. Include only attributes supported by '
    'the utterance. All values must be strings; use "yes"/"no" for booleans '
    'and comma-separated strings for lists. Use canonical platform names '
    '(PlayStation, Xbox, PC, Nintendo Switch) and ESRB labels such as '
    '"E (for Everyone)", "E 10+ (for Everyone 10 and Older)", '
    '"T (for Teen)", "M (for Mature)".'
)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalized_text(text):
    return ' '.join(text.lower().split())


def data(args, tokenizer):
    revision = DATASET_REVISION
    rows = {}
    file_hashes = {}
    for split in ('train', 'validation', 'test'):
        path = Path(hf_hub_download(DATASET, split + '.csv', repo_type='dataset', revision=revision))
        file_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows[split] = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
    # Audit against complete official test split before deterministic subsampling.
    seen = set()
    excluded = {}
    for split in ('test', 'validation', 'train'):
        before = len(rows[split])
        clean = []
        for row in rows[split]:
            text = normalized_text(row['ref'])
            if text not in seen:
                clean.append(row)
                seen.add(text)
        rows[split] = clean
        excluded[split] = before - len(clean)
    encoded = {}
    manifests = {}
    max_length = 0
    for split, limit in [('train', args.train_size), ('validation', args.val_size), ('test', args.test_size)]:
        selected = random.Random(2026).sample(rows[split], min(limit, len(rows[split])))
        encoded[split] = []
        manifests[split] = []
        for row in selected:
            mr = row['mr']
            target = {'dialogue_act': mr.split('(', 1)[0]}
            target.update(re.findall(r'(\w+)\[([^\]]*)\]', mr))
            assert set(target).issubset(ALLOWED_KEYS), (row['gem_id'], target)
            assert target['dialogue_act'] in ALLOWED_ACTS, (row['gem_id'], target)
            prompt = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['ref']}],
                tokenize=False, add_generation_prompt=True)
            answer = json.dumps(target, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            continuation = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)
            ids = prefix + continuation
            if len(ids) > args.max_length:
                raise ValueError(f'Example {row.get("gem_id")} has {len(ids)} tokens; no silent truncation')
            max_length = max(max_length, len(ids))
            item = {'id': row['gem_id'], 'utterance': row['ref'], 'target': target,
                    'prompt': prompt, 'input_ids': ids,
                    'labels': [-100] * len(prefix) + continuation}
            encoded[split].append(item)
            manifests[split].append({k: item[k] for k in ('id', 'utterance', 'target')})
    split_path = args.output / 'split_manifest.json'
    if split_path.exists() and json.loads(split_path.read_text()) != manifests:
        raise ValueError('Existing split manifest differs; use a fresh output directory')
    write_json(split_path, manifests)
    return encoded, {'dataset': DATASET, 'revision': revision, 'file_sha256': file_hashes,
                     'duplicate_utterances_excluded': excluded, 'sizes': {k: len(v) for k, v in encoded.items()},
                     'max_encoded_length': max_length,
                     'split_manifest_sha256': hashlib.sha256((args.output / 'split_manifest.json').read_bytes()).hexdigest()}


def batch(items, tokenizer):
    length = max(len(x['input_ids']) for x in items)
    ids = torch.full((len(items), length), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, item in enumerate(items):
        n = len(item['input_ids'])
        ids[i, :n] = torch.tensor(item['input_ids'])
        labels[i, :n] = torch.tensor(item['labels'])
        mask[i, :n] = 1
    return {k: v.cuda() for k, v in {'input_ids': ids, 'labels': labels, 'attention_mask': mask}.items()}


@torch.inference_mode()
def nll(model, items, tokenizer, batch_size):
    model.eval()
    total, tokens = 0.0, 0
    for i in range(0, len(items), batch_size):
        b = batch(items[i:i + batch_size], tokenizer)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(**b, use_cache=False)
        count = int((b['labels'][:, 1:] != -100).sum())
        total += out.loss.item() * count
        tokens += count
    return total / tokens


@torch.inference_mode()
def generate_metrics(model, items, tokenizer, batch_size, output_file):
    model.eval()
    valid, schema, exact, tp, predicted_slots, true_slots = 0, 0, 0, 0, 0, 0
    predictions = []
    tokenizer.padding_side = 'left'
    started = time.perf_counter()
    for start in range(0, len(items), batch_size):
        part = items[start:start + batch_size]
        inputs = tokenizer([x['prompt'] for x in part], padding=True, return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, use_cache=True)
        texts = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        for item, text in zip(part, texts):
            parsed = None
            try:
                parsed = json.loads(text.strip())
                valid += 1
            except (json.JSONDecodeError, ValueError):
                pass
            gold = item['target']
            is_schema = (isinstance(parsed, dict) and set(parsed).issubset(ALLOWED_KEYS)
                         and all(isinstance(v, str) for v in parsed.values())
                         and parsed.get('dialogue_act') in ALLOWED_ACTS)
            schema += int(is_schema)
            pred = parsed if is_schema else {}
            exact += int(pred == gold)
            gold_pairs, pred_pairs = set(gold.items()), set(pred.items())
            tp += len(gold_pairs & pred_pairs)
            predicted_slots += len(pred_pairs)
            true_slots += len(gold_pairs)
            predictions.append({'id': item['id'], 'utterance': item['utterance'], 'gold': gold, 'text': text, 'parsed': parsed, 'exact_match': pred == gold})
        print(json.dumps({'event': 'generation', 'done': min(start + batch_size, len(items)), 'total': len(items)}), flush=True)
    torch.cuda.synchronize()
    Path(output_file).write_text(''.join(json.dumps(p) + '\n' for p in predictions))
    return {'count': len(items), 'valid_json': valid / len(items), 'schema_valid': schema / len(items),
            'exact_match': exact / len(items), 'slot_micro_f1': 2 * tp / max(1, predicted_slots + true_slots),
            'generation_seconds': time.perf_counter() - started,
            'correct_slots': tp, 'predicted_slots': predicted_slots, 'gold_slots': true_slots}


def strip_adapters(model):
    for name, module in list(model.named_children()):
        if isinstance(module, AdapterLinear):
            base = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device='meta', dtype=module.weight.dtype)
            base.weight, base.bias = module.weight, module.bias
            setattr(model, name, base)
        else:
            strip_adapters(module)
    model.requires_grad_(False)


def frozen_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fit(model, method, seed, lr, steps, records, tokenizer, args, directory, evaluate_test=False):
    directory.mkdir(parents=True, exist_ok=True)
    strip_adapters(model)
    seed_all(seed)
    inject_adapters(model, method, rank=args.rank, targets=('q_proj', 'v_proj'))
    params = [p for p in model.parameters() if p.requires_grad]
    before = frozen_digest(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    order = []
    while len(order) < steps * args.batch_size:
        order.extend(torch.randperm(len(records['train']), generator=generator).tolist())
    logs = []
    best = math.inf
    best_state = None
    best_step = 0
    # Tiny warmup backward, then clear gradients: no optimizer/model state changes.
    warm = batch(records['train'][:args.batch_size], tokenizer)
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(**warm, use_cache=False).loss.backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (directory / 'training.jsonl').open('w') as log:
        for step in range(1, steps + 1):
            model.train()
            b = batch([records['train'][i] for i in order[(step - 1)*args.batch_size:step*args.batch_size]], tokenizer)
            ratio = min(1., step / max(1, steps * 0.1))
            for group in optimizer.param_groups:
                group['lr'] = lr * ratio
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**b, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError('Nonfinite adapter gradient')
            optimizer.step()
            record = {'step': step, 'loss': loss.item(), 'gradient_norm': float(grad), 'lr': lr * ratio}
            if step % args.eval_every == 0 or step == steps:
                record['validation_nll'] = nll(model, records['validation'], tokenizer, args.batch_size)
                if record['validation_nll'] < best:
                    best = record['validation_nll']
                    best_step = step
                    best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            log.write(json.dumps(record) + '\n'); log.flush()
            if step == 1 or step % 16 == 0 or step == steps:
                print(json.dumps({'event': 'train', 'method': method, 'seed': seed, 'steps': steps, **record}), flush=True)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    assert frozen_digest(model) == before, 'Frozen base weights changed'
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best_state[name])
    torch.save(best_state, directory / 'adapter.pt')
    result = {'method': method, 'seed': seed, 'lr': lr, 'rank': args.rank, 'steps': steps,
              'selected_step': best_step, 'validation_nll': best, 'trainable_parameters': sum(p.numel() for p in params),
              'train_and_validation_seconds': duration, 'peak_allocated_bytes': peak,
              'frozen_weights_unchanged': True, 'frozen_sha256': before}
    if evaluate_test:
        result['test_nll'] = nll(model, records['test'], tokenizer, args.batch_size)
        result['test'] = generate_metrics(model, records['test'], tokenizer, args.generation_batch_size, directory / 'predictions.jsonl')
    write_json(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/var/tmp/dora-bench/runs/extraction'))
    parser.add_argument('--train-size', type=int, default=1024)
    parser.add_argument('--val-size', type=int, default=128)
    parser.add_argument('--test-size', type=int, default=256)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--generation-batch-size', type=int, default=16)
    parser.add_argument('--pilot-steps', type=int, default=48)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[1e-4,3e-4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records, data_provenance = data(args, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.requires_grad_(False)
    manifest = {'model': MODEL, 'model_revision': MODEL_REVISION, 'dataset': data_provenance,
                'configuration': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                'gpu': torch.cuda.get_device_name(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
                'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['experiments/adapters.py', 'experiments/extraction.py', 'dora.py']},
                'system_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
                'protocol': 'Official ViGGO splits, query-text deduplication, fixed deterministic subsets. Two equally budgeted LR pilots per method selected by validation NLL; final checkpoint by validation NLL. Test greedy generation only after selection. No adapter dropout; rank8 effective scale1; A Kaiming and B zero. Target q_proj/v_proj, frozen BF16 base, FP32 adapters, BF16 autocast.'}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing run differs in configuration, source, data or environment; use a fresh output directory')
    if not manifest_path.exists() and (args.output / 'baseline.json').exists():
        raise ValueError('Cached baseline lacks provenance; use a fresh output directory')
    write_json(manifest_path, manifest)
    baseline_file = args.output / 'baseline.json'
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
    else:
        baseline = {'method': 'baseline', 'test_nll': nll(model, records['test'], tokenizer, args.batch_size),
                    'test': generate_metrics(model, records['test'], tokenizer, args.generation_batch_size, args.output / 'baseline_predictions.jsonl')}
        write_json(baseline_file, baseline)
    print(json.dumps({'event':'baseline','result':baseline}),flush=True)
    pilots = []
    selected = {}
    for method in METHODS:
        for lr in args.learning_rates:
            directory = args.output / f'pilot_{method}_{lr:g}'
            if (directory / 'result.json').exists():
                result=json.loads((directory/'result.json').read_text())
            else:
                result=fit(model,method,args.seeds[0],lr,args.pilot_steps,records,tokenizer,args,directory)
            pilots.append(result)
        selected[method] = min((p for p in pilots if p['method']==method),key=lambda x:x['validation_nll'])['lr']
    write_json(args.output/'tuning.json',{'pilots':pilots,'selected_learning_rates':selected})
    results=[]
    for seed in args.seeds:
        for method in METHODS:
            directory=args.output/f'{method}_seed{seed}'
            if (directory/'result.json').exists():
                result=json.loads((directory/'result.json').read_text())
            else:
                result=fit(model,method,seed,selected[method],args.steps,records,tokenizer,args,directory,True)
            results.append(result)
            write_json(args.output/'results.json',{'manifest':manifest,'baseline':baseline,'tuning':pilots,'selected_learning_rates':selected,'results':results})
            print(json.dumps({'event':'completed','result':result}),flush=True)


if __name__=='__main__':
    main()
