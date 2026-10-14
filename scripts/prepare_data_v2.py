#!/usr/bin/env python3
"""Prepare 20,000 intact MetaMathQA examples with disjoint problem groups."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from rankstudy.v2.data import (QuestionOverlap, connected_groups, digest, encode_rows,
                               normalize, stratified_take)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data"))
    parser.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--out", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data_v2"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data_v2_manifest.json"))
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if (args.out / "manifest.json").exists():
        raise SystemExit("Dataset already sealed; use a fresh output path for any new preparation.")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.data_root / "meta-math_MetaMathQA/MetaMathQA-395K.json"
    rows = json.loads(source.read_text())
    # Only question columns leave this dedicated deduplication preparation.
    questions = pq.read_table(args.data_root / "openai_gsm8k/main/test-00000-of-00001.parquet",
                              columns=["question"]).column("question").to_pylist()
    with (args.data_root / "HuggingFaceH4_MATH-500/test.jsonl").open() as f:
        questions += [json.loads(line)["problem"] for line in f if line.strip()]
    overlap = QuestionOverlap(questions)
    groups = connected_groups(rows)
    blocked = {}
    alias_cache = {}
    group_ids = {}
    group_sources = {}
    for gi, indices in enumerate(groups):
        aliases = sorted({normalize(rows[i].get(k) or "") for i in indices
                          for k in ("original_question", "query")} - {""})
        group_id = digest("\n".join(aliases))
        group_sources[group_id] = Counter((rows[i].get("type") or "unknown").split("_")[0]
                                         for i in indices).most_common(1)[0][0]
        for text in aliases:
            if text not in alias_cache:
                alias_cache[text] = overlap.match(text)
            if alias_cache[text]:
                blocked[group_id] = alias_cache[text]
                break
        for i in indices:
            group_ids[i] = group_id
        if gi % 1000 == 0:
            print(json.dumps({"event": "dedup", "groups_checked": gi, "blocked_groups": len(blocked)}), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    candidates, excluded, seen = [], Counter(), set()
    for i, r in enumerate(rows):
        gid = group_ids[i]
        if gid in blocked:
            excluded["heldout_overlap_rows"] += 1
            continue
        if not (r.get("query") or "").strip() or not (r.get("response") or "").strip():
            excluded["empty_rows"] += 1
            continue
        qkey = normalize(r["query"])
        if qkey in seen:
            excluded["duplicate_query_rows"] += 1
            continue
        seen.add(qkey)
        candidates.append(i)

    eligible, length_hist = [], Counter()
    for start in range(0, len(candidates), 512):
        inds = candidates[start:start + 512]
        texts = [f"Question: {rows[i]['query']}\nAnswer: {rows[i]['response']}" for i in inds]
        tokenized = tok(texts, add_special_tokens=False)["input_ids"]
        for i, ids in zip(inds, tokenized):
            length_hist[(len(ids) + 1) // 256] += 1
            if len(ids) + 1 > 2048:
                excluded["overlength_rows"] += 1
            else:
                eligible.append(i)
        if start % (512 * 40) == 0:
            print(json.dumps({"event": "length_filter", "rows_checked": start, "eligible": len(eligible)}), flush=True)

    by_group = defaultdict(list)
    for i in eligible:
        by_group[group_ids[i]].append(i)
    # Reserve GROUPS before sampling training EXAMPLES. Repeated groups within
    # training are allowed; the entire component is excluded from other pools.
    tune_g, rest = stratified_take(list(by_group), 512, args.seed, lambda g: group_sources[g])
    audit_g, train_g = stratified_take(rest, 512, args.seed + 1, lambda g: group_sources[g])
    train_candidates = [i for g in train_g for i in by_group[g]]
    train, _ = stratified_take(train_candidates, 20000, args.seed + 2, lambda i: rows[i]["type"])
    representative = lambda g: min(by_group[g], key=lambda i: digest(f"{args.seed}:representative:{i}"))
    splits = {"train": train, "dev_tune": [representative(g) for g in tune_g],
              "dev_audit": [representative(g) for g in audit_g]}
    split_stats = {}
    for name, indices in splits.items():
        encoded_rows = []
        for start in range(0, len(indices), 256):
            raw = [{"source_index": i, "group_id": group_ids[i], "source_type": rows[i]["type"],
                    "query": rows[i]["query"], "response": rows[i]["response"]}
                   for i in indices[start:start + 256]]
            encoded_rows += encode_rows(raw, tok)
        assert all(row is not None for row in encoded_rows)
        out = args.out / f"{name}.jsonl"
        with out.open("w") as f:
            for r in encoded_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        split_stats[name] = {"rows": len(indices), "groups": len({group_ids[i] for i in indices}),
                             "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                             "input_tokens": sum(r["input_tokens"] for r in encoded_rows),
                             "supervised_tokens": sum(r["supervised_tokens"] for r in encoded_rows),
                             "types": dict(Counter(rows[i]["type"] for i in indices))}
    memberships = {n: set(group_ids[i] for i in ids) for n, ids in splits.items()}
    assert not (memberships["train"] & memberships["dev_tune"] or memberships["train"] & memberships["dev_audit"]
                or memberships["dev_tune"] & memberships["dev_audit"])
    # Probe is training-side only; group-disjoint from both development pools.
    probe_groups, _ = stratified_take(sorted(memberships["train"]), 64, args.seed + 3,
                                      lambda g: group_sources[g])
    probe_indices = [min(i for i in train if group_ids[i] == g) for g in probe_groups]
    (args.out / "probe_source_indices.json").write_text(json.dumps(probe_indices) + "\n")
    manifest = {"protocol": "adaptive-rank-v2", "seed": args.seed, "source_rows": len(rows),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_revisions": json.loads(Path("artifacts/hf_revisions.json").read_text()),
                "tokenizer": args.model, "max_length": 2048, "output": str(args.out),
                "original_groups": len(groups), "eligible_groups": len(by_group),
                "eligible_rows": len(eligible), "train_candidate_rows": len(train_candidates),
                "blocked_groups": dict(Counter(blocked.values())), "excluded": dict(excluded),
                "splits": split_stats, "pre_filter_length_bins_256": dict(length_hist),
                "holdout_question_sha256": digest("\n".join(sorted(map(normalize, questions)))),
                "near_duplicate_rule": "equal numeral multiset AND word/punctuation trigram Jaccard >= 0.85",
                "template": "Question: {query}\nAnswer: {response}<EOS>",
                "boundary_rule": "full-text offsets; response-overlapping whitespace token supervised; explicit EOS",
                "limitations": ["Conservative near-duplicate detection does not detect all paraphrases.",
                                "Base-model pretraining contamination is not measured."]}
    content = json.dumps(manifest, indent=2) + "\n"
    (args.out / "manifest.json").write_text(content)
    args.manifest.write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
