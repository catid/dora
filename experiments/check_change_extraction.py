"""Full Qwen/ViGGO endpoint control through five clipped AdamW updates."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import change_ablations as ab
from experiments import change_extraction as study
from experiments import extraction as task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    config = json.loads(Path('/var/tmp/dora-bench/original_bench/extraction/manifest.json').read_text())['configuration']
    config.update(output=args.output, eval_every=5)
    config = SimpleNamespace(**config)
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records, provenance = task.data(config, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(task.MODEL, revision=task.MODEL_REVISION,
                                              dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    backend = study.backend()
    checks = []
    for reference, recipe in [('dora_original', 'ab_original'), ('dora', 'ab_revised')]:
        task.strip_adapters(model, backend.adapter_types)
        reference_dir, recipe_dir = args.output / reference, args.output / recipe
        old = task.fit(model, reference, 42, .0003, 5, records, tokenizer, config, reference_dir)
        new = task.fit(model, recipe, 42, .0003, 5, records, tokenizer, config, recipe_dir, backend=backend)
        for field in ('selected_step', 'validation_nll', 'trainable_parameters', 'frozen_sha256'):
            assert old[field] == new[field], (recipe, field, old[field], new[field])
        old_log, new_log = [(d / 'training.jsonl').read_bytes() for d in (reference_dir, recipe_dir)]
        assert old_log == new_log, f'{recipe}: loss/gradient trajectory differs'
        old_state = torch.load(reference_dir / 'adapter.pt', weights_only=True)
        new_state = torch.load(recipe_dir / 'adapter.pt', weights_only=True)
        count = 0
        for name, value in old_state.items():
            target_name = name
            if reference == 'dora_original':
                prefix, suffix = name.rsplit('.', 1)
                target_name = prefix + '.' + {'lora_A': 'lora_B', 'lora_B': 'lora_A'}.get(suffix, suffix)
            assert torch.equal(value, new_state[target_name]), (recipe, name)
            count += 1
        checks.append({'reference': reference, 'recipe': recipe, 'optimizer_steps': 5,
                       'training_log_bitwise_equal': True, 'checkpoint_tensors_bitwise_equal': count,
                       'validation_nll_bitwise_equal': True, 'frozen_sha256': old['frozen_sha256']})
    sources = ['experiments/change_ablations.py', 'experiments/change_extraction.py',
               'experiments/check_change_extraction.py', 'experiments/extraction.py',
               'experiments/adapters.py', 'experiments/original_dora.py', 'dora_row.py']
    task.write_json(args.output / 'control.json', {
        'status': 'pass', 'checks': checks, 'dataset': provenance,
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}})
    for name in sources:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    print(json.dumps({'event': 'control_passed', 'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
