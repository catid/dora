"""Matched PEFT domain-retrieval experiment on the official SciFact split.

Run from the repository root, for example:
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/var/tmp/dora-bench/cache \
    python -m experiments.retrieval --seeds 42 43 44

Every method gets the same validation-only learning-rate grid and splits.
Test scores are never used for checkpoint or hyperparameter selection.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATASET = "BeIR/scifact"
DATASET_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
QRELS = "BeIR/scifact-qrels"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(args):
    def download(repo, name, revision):
        return hf_hub_download(repo, name, repo_type="dataset", revision=revision)

    paths = {
        "corpus": download(DATASET, "corpus/corpus-00000-of-00001.parquet", DATASET_REVISION),
        "queries": download(DATASET, "queries/queries-00000-of-00001.parquet", DATASET_REVISION),
        "train_qrels": download(QRELS, "train.tsv", QRELS_REVISION),
        "test_qrels": download(QRELS, "test.tsv", QRELS_REVISION),
    }
    corpus = {
        str(row["_id"]): (row["title"] + "\n" + row["text"]).strip()
        for row in pq.read_table(paths["corpus"]).to_pylist()
    }
    queries = {str(row["_id"]): row["text"] for row in pq.read_table(paths["queries"]).to_pylist()}

    def relevance(path):
        result = defaultdict(dict)
        with open(path) as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if int(row["score"]) > 0:
                    result[row["query-id"]][row["corpus-id"]] = int(row["score"])
        return dict(result)

    train_relevance = relevance(paths["train_qrels"])
    test_relevance = relevance(paths["test_qrels"])
    assert set(train_relevance).isdisjoint(test_relevance)
    # The public split uses query IDs. Also prevent exact normalized text overlap.
    normalize = lambda text: " ".join(text.lower().split())
    test_texts = {normalize(queries[qid]) for qid in test_relevance}
    duplicate_ids = sorted(qid for qid in train_relevance if normalize(queries[qid]) in test_texts)
    eligible = sorted(set(train_relevance) - set(duplicate_ids))
    random.Random(args.split_seed).shuffle(eligible)
    num_valid = max(1, round(args.validation_fraction * len(eligible)))
    valid_ids, train_ids = sorted(eligible[:num_valid]), sorted(eligible[num_valid:])
    # Keep exact duplicate training claims together if they exist in the source.
    valid_texts = {normalize(queries[qid]) for qid in valid_ids}
    overlap = [qid for qid in train_ids if normalize(queries[qid]) in valid_texts]
    train_ids = sorted(set(train_ids) - set(overlap))
    valid_ids = sorted(valid_ids + overlap)
    splits = {"train": train_ids, "validation": valid_ids, "test": sorted(test_relevance)}
    assert not (set(train_ids) & set(valid_ids) or set(train_ids) & set(test_relevance))
    for qid, documents in (train_relevance | test_relevance).items():
        assert qid in queries and all(doc in corpus for doc in documents)
    provenance = {
        "dataset": DATASET, "revision": DATASET_REVISION,
        "qrels": QRELS, "qrels_revision": QRELS_REVISION,
        "source_sha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in paths.items()},
        "corpus_size": len(corpus), "official_train_query_count": len(train_relevance),
        "query_counts": {key: len(value) for key, value in splits.items()},
        "excluded_train_ids_with_exact_test_text": duplicate_ids,
        "split_seed": args.split_seed,
        "leakage_scope": "Official query split plus exact normalized-text deduplication. Shared relevant documents and paraphrased claims may occur across the official split; pretrained-data contamination is not audited.",
    }
    return corpus, queries, train_relevance | test_relevance, splits, provenance


def tokenize_all(tokenizer, texts, maximum):
    ids = sorted(texts)
    encoded = tokenizer([texts[key] for key in ids], truncation=True, max_length=maximum, padding=False)
    return {key: {name: values[index] for name, values in encoded.items()} for index, key in enumerate(ids)}


def make_batch(tokenizer, tokenized, ids, device):
    return tokenizer.pad(
        [tokenized[key] for key in ids], padding=True, pad_to_multiple_of=8, return_tensors="pt"
    ).to(device)


def embeddings(model, batch):
    hidden = model(**batch).last_hidden_state.float()
    mask = batch["attention_mask"].unsqueeze(-1)
    return F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1), dim=-1)


@torch.inference_mode()
def encode(model, tokenizer, tokenized, ids, batch_size, device):
    # Length bucketing reduces padding without changing ID order in the output.
    ordered = sorted(range(len(ids)), key=lambda i: len(tokenized[ids[i]]["input_ids"]))
    result = torch.empty(len(ids), model.config.hidden_size, device=device, dtype=torch.float32)
    for start in range(0, len(ids), batch_size):
        positions = ordered[start:start + batch_size]
        batch = make_batch(tokenizer, tokenized, [ids[i] for i in positions], device)
        result[positions] = embeddings(model, batch)
    return result


def query_metrics(ranked, relevant):
    hits = [relevant.get(docid, 0) for docid in ranked]
    recall = sum(value > 0 for value in hits) / len(relevant)
    reciprocal = next((1 / (i + 1) for i, value in enumerate(hits) if value > 0), 0.0)
    dcg = sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(hits))
    ideal = sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(sorted(relevant.values(), reverse=True)[:10]))
    return {"recall_at_10": recall, "mrr_at_10": reciprocal, "ndcg_at_10": dcg / ideal}


@torch.inference_mode()
def evaluate(model, tokenizer, corpus_tokens, query_tokens, relevance, splits, args, run_dir,
             evaluation_splits=("validation", "test")):
    model.eval()
    corpus_ids = sorted(corpus_tokens)
    warmup_ids = sorted(corpus_ids, key=lambda key: len(corpus_tokens[key]["input_ids"]))[-args.eval_batch_size:]
    for _ in range(3):
        embeddings(model, make_batch(tokenizer, corpus_tokens, warmup_ids, args.device))
    torch.cuda.synchronize()
    started = time.perf_counter()
    corpus_embeddings = encode(model, tokenizer, corpus_tokens, corpus_ids, args.eval_batch_size, args.device)
    torch.cuda.synchronize()
    corpus_seconds = time.perf_counter() - started
    result = {"corpus_encoding_seconds": corpus_seconds, "corpus_documents_per_second": len(corpus_ids) / corpus_seconds}
    for split in evaluation_splits:
        ids = splits[split]
        started = time.perf_counter()
        query_embeddings = encode(model, tokenizer, query_tokens, ids, args.eval_batch_size, args.device)
        scores = query_embeddings @ corpus_embeddings.T
        best_scores, best_positions = scores.topk(10, dim=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        details = []
        for qid, values, positions in zip(ids, best_scores.cpu().tolist(), best_positions.cpu().tolist()):
            ranked = [corpus_ids[position] for position in positions]
            details.append({"query_id": qid, "top_10": ranked, "scores": values, **query_metrics(ranked, relevance[qid])})
        result[split] = {metric: statistics.mean(row[metric] for row in details) for metric in ("recall_at_10", "mrr_at_10", "ndcg_at_10")}
        result[split]["query_count"] = len(ids)
        result[split]["query_encoding_and_ranking_seconds"] = elapsed
        write_json(run_dir / f"{split}_per_query.json", details)
    torch.cuda.synchronize()
    result["total_seconds"] = corpus_seconds + sum(result[split]["query_encoding_and_ranking_seconds"] for split in evaluation_splits)
    return result


def train(model, tokenizer, corpus_tokens, query_tokens, relevance, splits, args, seed, run_dir):
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(splits["train"]) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * args.warmup_fraction))
    generator = random.Random(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    started = time.perf_counter()
    step = 0
    epoch_summaries = []
    with (run_dir / "training.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            query_ids = splits["train"].copy()
            generator.shuffle(query_ids)
            losses = []
            for start in range(0, len(query_ids), args.batch_size):
                ids = query_ids[start:start + args.batch_size]
                docs = [generator.choice(sorted(relevance[qid])) for qid in ids]
                positive_mask = torch.tensor([[doc in relevance[qid] for doc in docs] for qid in ids], device=args.device)
                step += 1
                if step <= warmup_steps:
                    factor = step / warmup_steps
                else:
                    factor = (total_steps - step + 1) / max(1, total_steps - warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * factor
                optimizer.zero_grad(set_to_none=True)
                query_embeddings = embeddings(model, make_batch(tokenizer, query_tokens, ids, args.device))
                document_embeddings = embeddings(model, make_batch(tokenizer, corpus_tokens, docs, args.device))
                scores = query_embeddings @ document_embeddings.T / args.temperature
                loss = (scores.logsumexp(dim=1) - scores.masked_fill(~positive_mask, -torch.inf).logsumexp(dim=1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Nonfinite gradient at step {step}")
                optimizer.step()
                record = {"epoch": epoch + 1, "step": step, "loss": loss.item(), "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": grad_norm.item(), "elapsed_seconds": time.perf_counter() - started}
                log.write(json.dumps(record) + "\n")
                log.flush()
                losses.append(loss.item())
            epoch_summary = {"epoch": epoch + 1, "mean_loss": statistics.mean(losses), "elapsed_seconds": time.perf_counter() - started}
            epoch_summaries.append(epoch_summary)
            print(json.dumps({"event": "epoch", "method": args.current_method, "seed": seed, **epoch_summary}), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "steps": step, "epochs": epoch_summaries, "seconds": seconds,
        "queries_seen": len(splits["train"]) * args.epochs,
        "queries_per_second": len(splits["train"]) * args.epochs / seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline_memory,
    }
    torch.save({name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}, run_dir / "adapter.pt")
    return result


def provenance(args, dataset):
    root = Path(__file__).resolve().parent.parent
    source = [Path(__file__).resolve(), root / "experiments" / "adapters.py"]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.pop("current_method", None)
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "python": sys.version, "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "huggingface_hub", "numpy", "pyarrow")},
        "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source if path.exists()},
        "model": {"repository": MODEL, "revision": MODEL_REVISION, "pooling": "attention-mask mean pooling; L2 normalization", "frozen_dtype": "bfloat16", "adapter_dtype": "float32"},
        "dataset": dataset, "configuration": config,
        "protocol": {
            "target_modules": "BERT self-attention query and value projections in all 6 layers",
            "alpha_over_rank": 1, "adapter_dropout": 0,
            "objective": "Supervised in-batch contrastive loss; all known relevant documents in the batch contribute to the positive numerator.",
            "checkpoint_selection": "Fixed final epoch. Equal per-method learning-rate grid selected by validation nDCG@10 on the first seed; other seeds use that selected rate. Ties select the smaller rate. No test tuning.",
            "inference_timing": "Unmerged adapters; three full-length batch warmups before synchronized corpus encoding. Includes host token-padding and transfer.",
            "ranking": "Exhaustive cosine similarity over all 5183 corpus documents; no approximate index.",
            "metrics": "Macro query mean Recall@10, reciprocal rank truncated at 10, and nDCG@10 with gain 2^relevance-1.",
            "limitations": "Small single-domain adaptation set and one encoder; confidence intervals measure query and seed variability, not broad task generalization.",
        },
    }


def summarize(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)
    result = {}
    for method, records in grouped.items():
        metrics = {}
        for metric in ("recall_at_10", "mrr_at_10", "ndcg_at_10"):
            values = [record["evaluation"]["test"][metric] for record in records]
            metrics[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None, "per_seed": values}
        result[method] = {"completed_seeds": [record["seed"] for record in records], "test": metrics}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/var/tmp/dora-bench/runs/retrieval"))
    parser.add_argument("--methods", nargs="+", choices=("frozen", "lora", "dora", "nora", "dora_nora"), default=["frozen", "lora", "dora", "nora", "dora_nora"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 1e-3])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20261008)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("This measured experiment requires a CUDA device")
    for name in ("rank", "epochs", "batch_size", "eval_batch_size", "max_length", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not (0 < args.validation_fraction < 1 and 0 < args.warmup_fraction < 1 and args.temperature > 0):
        parser.error("fractions must lie between zero and one, and temperature must be positive")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus, queries, relevance, splits, dataset_provenance = load_data(args)
    write_json(args.output_dir / "splits.json", splits)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    corpus_tokens = tokenize_all(tokenizer, corpus, args.max_length)
    query_tokens = tokenize_all(tokenizer, queries, args.max_length)
    metadata = provenance(args, dataset_provenance)
    write_json(args.output_dir / "provenance.json", metadata)
    results, tuning = [], {}

    def new_model(method, seed):
        seed_everything(seed)
        model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
        seed_everything(seed)
        if method == "frozen":
            model.requires_grad_(False)
        else:
            from experiments.adapters import inject_adapters
            inject_adapters(model, method=method, rank=args.rank, targets=("query", "value"))
        return model

    def run_record(model, method, seed):
        record = {"method": method, "seed": seed,
                  "total_parameters": sum(p.numel() for p in model.parameters()),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad]}
        print(json.dumps({"event": "start", **record}), flush=True)
        return record

    def save_results():
        write_json(args.output_dir / "results.json", {"provenance": metadata, "tuning": tuning, "runs": results, "summary": summarize(results)})

    for method in args.methods:
        args.current_method = method
        if method != "frozen":
            candidates = []
            for learning_rate in sorted(set(args.learning_rates)):
                args.learning_rate = learning_rate
                seed = args.seeds[0]
                run_dir = args.output_dir / "tuning" / f"{method}_lr{learning_rate:g}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                model = new_model(method, seed)
                record = run_record(model, method, seed)
                record.update(learning_rate=learning_rate, artifact_directory=str(run_dir))
                record["training"] = train(model, tokenizer, corpus_tokens, query_tokens, relevance, splits, args, seed, run_dir)
                record["evaluation"] = evaluate(model, tokenizer, corpus_tokens, query_tokens, relevance, splits, args, run_dir, ("validation",))
                write_json(run_dir / "result.json", record)
                candidates.append(record)
                print(json.dumps({"event": "tuning_result", "method": method, "learning_rate": learning_rate, "validation": record["evaluation"]["validation"]}), flush=True)
                del model
                torch.cuda.empty_cache()
            winner = max(candidates, key=lambda record: (record["evaluation"]["validation"]["ndcg_at_10"], -record["learning_rate"]))
            tuning[method] = {"selected_learning_rate": winner["learning_rate"], "selection_metric": "validation.ndcg_at_10", "candidates": candidates}
            args.learning_rate = winner["learning_rate"]
            save_results()
        for seed in args.seeds[:1] if method == "frozen" else args.seeds:
            run_dir = args.output_dir / f"{method}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model = new_model(method, seed)
            record = run_record(model, method, seed)
            if method != "frozen":
                record["learning_rate"] = args.learning_rate
                if seed == args.seeds[0]:
                    checkpoint = Path(winner["artifact_directory"]) / "adapter.pt"
                    state = torch.load(checkpoint, map_location=args.device, weights_only=True)
                    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                    assert set(state) == expected
                    model.load_state_dict(state, strict=False)
                    record["training"] = winner["training"]
                    record["reused_tuning_checkpoint"] = str(checkpoint)
                    shutil.copy2(checkpoint, run_dir / "adapter.pt")
                    shutil.copy2(checkpoint.parent / "training.jsonl", run_dir / "training.jsonl")
                else:
                    record["training"] = train(model, tokenizer, corpus_tokens, query_tokens, relevance, splits, args, seed, run_dir)
            record["evaluation"] = evaluate(model, tokenizer, corpus_tokens, query_tokens, relevance, splits, args, run_dir)
            write_json(run_dir / "result.json", record)
            results.append(record)
            save_results()
            print(json.dumps({"event": "result", "method": method, "seed": seed, "test": record["evaluation"]["test"], "training_seconds": record.get("training", {}).get("seconds")}), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
