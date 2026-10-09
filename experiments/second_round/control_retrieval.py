"""Quantify BF16 zero-update adapter drift versus the native frozen encoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

from experiments.retrieval import MODEL, MODEL_REVISION, embeddings, evaluate, make_batch, seed_everything, tokenize_all
from experiments.second_round.retrieval import METHODS, load_data
from experiments.second_round_adapters import inject_adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = json.loads((args.root / 'results.json').read_text())
    config = SimpleNamespace(**result['provenance']['configuration'])
    torch.cuda.set_device(config.device)
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision('highest')
    data = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    for dataset in data.values():
        dataset['corpus_tokens'] = tokenize_all(tokenizer, dataset['corpus'], config.max_length)
        dataset['query_tokens'] = tokenize_all(tokenizer, dataset['queries'], config.max_length)
    probe_ids = sorted(data['nfcorpus']['corpus'])[:128]
    probe = make_batch(tokenizer, data['nfcorpus']['corpus_tokens'], probe_ids, config.device)
    rows, reference = [], None
    with torch.inference_mode():
        for method in ['frozen', *METHODS]:
            seed_everything(config.seeds[0])
            model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation='sdpa').to(config.device)
            seed_everything(config.seeds[0])
            if method != 'frozen':
                inject_adapters(model, method, rank=config.ranks[0], targets=('query', 'value'))
            model.eval()
            encoded = embeddings(model, probe)
            if reference is None:
                reference = encoded.clone()
            row = {'method': method, 'rank': 0 if method == 'frozen' else config.ranks[0],
                   'max_absolute_probe_embedding_difference': (encoded - reference).abs().max().item(),
                   'mean_probe_cosine_similarity': (encoded * reference).sum(dim=1).mean().item(),
                   'evaluation': {}}
            for name, ds in data.items():
                directory = args.root / 'initial_control' / method / name
                directory.mkdir(parents=True, exist_ok=True)
                row['evaluation'][name] = evaluate(model, tokenizer, ds['corpus_tokens'], ds['query_tokens'], ds['relevance'], ds['splits'], config, directory, ('test',))
            rows.append(row)
            del model
            torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
    output = {'protocol': 'No training; all adapters have zero B. Same first rank and seed as main protocol. Native BF16 Linear fuses bias while adapters add bias after projection, introducing a possible rounding difference. Probe uses first128 sorted NFCorpus documents; full test metrics/rankings retained for both datasets.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'runs': rows}
    (args.root / 'initial_adapter_control.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
