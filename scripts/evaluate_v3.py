#!/usr/bin/env python3
"""Deterministic greedy generation and answer-parser audit runner.

The default protocol is 2,048 new tokens.  ``--limit`` is intended for public
TRAIN preflight only; official test runs are created only after the selection
freeze and remain sealed until then.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rankstudy.v3.answer_parser import equivalent, parse_candidate
from rankstudy.v3.adapters import inject
from rankstudy.v3.checkpoint import load_adapter


class ParseTimeout(TimeoutError):
    pass


def _alarm(_signum, _frame):
    raise ParseTimeout("parser timeout")


def parse_with_timeout(prediction, reference, dataset, timeout_s):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        ok, detail = equivalent(prediction, reference, dataset)
        parsed = parse_candidate(prediction, dataset)
        return ok, detail, parsed
    except ParseTimeout as exc:
        return False, {"mode": "timeout", "error": str(exc)}, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def load_rows(path, dataset, limit):
    if path.suffix == ".parquet":
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    output = []
    for index, row in enumerate(rows[:limit] if limit else rows):
        question = row.get("question", row.get("problem", row.get("query")))
        reference = row.get("answer", row.get("response"))
        if not isinstance(question, str) or not isinstance(reference, str):
            raise ValueError(f"row {index} has no question/answer text")
        output.append({"index": index, "question": question, "reference": reference,
                       "source_id": row.get("unique_id", row.get("source_index", index))})
    return output


def load_model(model_path, checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", local_files_only=True).to(device)
    model.config.use_cache = True
    if checkpoint:
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        modules = manifest["modules"]
        first = next(iter(modules.values()))
        pattern = {name: entry["rank"] for name, entry in modules.items()}
        inject(model, max(pattern.values()), first["init"], float(first["scale"]), 0,
               first["score"], ("q_proj", "o_proj"), pattern)
        load_adapter(checkpoint, model, str(device))
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_one(model, tokenizer, question, device, max_new_tokens):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    start = time.perf_counter()
    output = model.generate(**inputs, do_sample=False, num_beams=1,
                            max_new_tokens=max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True)
    elapsed = time.perf_counter() - start
    generated = output[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    ended = bool(generated.numel() and generated[-1].item() == tokenizer.eos_token_id)
    return text, int(generated.numel()), ended, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--parser-timeout-s", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_new_tokens != 2048 and not args.limit:
        raise ValueError("non-default generation length requires a bounded preflight")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_model(args.model, args.checkpoint, args.device)
    rows = load_rows(args.input, args.dataset, args.limit)
    records = []
    for row in rows:
        try:
            generated, token_count, ended, elapsed = generate_one(model, tokenizer, row["question"], args.device, args.max_new_tokens)
            parsed_ok, detail, parsed = parse_with_timeout(generated, row["reference"], args.dataset, args.parser_timeout_s)
            error = None
        except Exception as exc:
            generated, token_count, ended, elapsed, parsed_ok, detail, parsed = "", 0, False, 0., False, {"mode": "generation_error", "error": repr(exc)}, None
            error = repr(exc)
        records.append({**row, "prediction": generated, "generated_tokens": token_count,
                        "ended_with_eos": ended, "truncated": token_count >= args.max_new_tokens and not ended,
                        "generation_seconds": elapsed, "correct": parsed_ok, "parser": detail,
                        "parser_supported": None if parsed is None else parsed.supported, "error": error})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "rankstudy-v3-generation", "model": args.model,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "dataset": args.dataset, "rows": len(records), "max_new_tokens": args.max_new_tokens,
              "greedy": True, "parser_timeout_s": args.parser_timeout_s,
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "records": records,
              "summary": {"correct": sum(r["correct"] for r in records),
                          "parser_failures": sum(not r["parser_supported"] for r in records),
                          "timeouts": sum(r["parser"].get("mode") == "timeout" for r in records),
                          "truncated": sum(r["truncated"] for r in records),
                          "generation_errors": sum(r["error"] is not None for r in records)}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__": main()
