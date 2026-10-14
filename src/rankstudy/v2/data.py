"""Grouped query/response preparation and immutable effective-update manifests."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def connected_groups(rows):
    """Union ALL original/query aliases, never substitute original for query."""
    parents = list(range(len(rows)))
    seen = {}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        for field in ("original_question", "query"):
            text = normalize(row.get(field) or "")
            if not text:
                continue
            if text in seen:
                a, b = root(i), root(seen[text])
                parents[max(a, b)] = min(a, b)
            else:
                seen[text] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    return list(groups.values())


def shingles(text):
    tokens = re.findall(r"\w+|[^\w\s]", normalize(text))
    grams = {tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))}
    return grams, tuple(sorted(re.findall(r"\d+(?:\.\d+)?", normalize(text))))


class QuestionOverlap:
    """Conservative question-only decontamination with a declared exact rule.

    Near duplicates require equal numeral multisets and >=0.85 Jaccard of
    word/punctuation trigrams. All potential matches sharing a trigram are
    examined, so retrieval is not a heuristic additional filter. This cannot
    detect arbitrary paraphrases or establish no pretraining contamination.
    """
    def __init__(self, questions, threshold=0.85):
        self.threshold = threshold
        self.exact = {normalize(q) for q in questions}
        self.features = [shingles(q) for q in questions]
        self.index = defaultdict(set)
        for i, (grams, _) in enumerate(self.features):
            for gram in grams:
                self.index[gram].add(i)

    def match(self, text):
        if normalize(text) in self.exact:
            return "normalized_exact"
        grams, numbers = shingles(text)
        candidates = Counter(i for g in grams for i in self.index.get(g, ()))
        for i, intersection in candidates.items():
            other, other_numbers = self.features[i]
            if numbers == other_numbers and intersection / len(grams | other) >= self.threshold:
                return "near_trigram_jaccard"
        return None


def encode_rows(rows, tokenizer, max_length=2048):
    """Tokenize the literal full text with offsets; append exactly one EOS.

    A boundary-spanning token is supervised only when the prompt portion is
    whitespace. This includes BPE tokens encoding the leading response space.
    Otherwise fail instead of silently supervising prompt content.
    """
    prompts = [f"Question: {r['query']}\nAnswer: " for r in rows]
    texts = [p + r["response"] for p, r in zip(prompts, rows)]
    encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    output = []
    for row, prompt, ids, offsets in zip(rows, prompts, encoded.input_ids, encoded.offset_mapping):
        if len(ids) + 1 > max_length:
            output.append(None)
            continue
        labels = []
        for token, (lo, hi) in zip(ids, offsets):
            if lo < len(prompt) < hi and prompt[lo:].strip():
                raise ValueError("BPE token crosses a non-whitespace response boundary")
            labels.append(token if hi > len(prompt) else -100)
        output.append({**row, "input_ids": ids + [tokenizer.eos_token_id],
                       "labels": labels + [tokenizer.eos_token_id],
                       "input_tokens": len(ids) + 1,
                       "supervised_tokens": sum(x != -100 for x in labels) + 1})
    return output


def stratified_take(items, n, seed, stratum):
    if len(items) < n:
        raise ValueError(f"requested {n} rows from {len(items)}")
    groups = defaultdict(list)
    for item in items:
        groups[stratum(item)].append(item)
    rng = random.Random(seed)
    keys = sorted(groups)
    quotas = {k: n * len(groups[k]) // len(items) for k in keys}
    for k in sorted(keys, key=lambda k: (-(n * len(groups[k]) % len(items)), k))[:n - sum(quotas.values())]:
        quotas[k] += 1
    chosen, rest = [], []
    for k in keys:
        rng.shuffle(groups[k])
        chosen.extend(groups[k][:quotas[k]])
        rest.extend(groups[k][quotas[k]:])
    rng.shuffle(chosen)
    return chosen, rest


def effective_manifest(rows, seed, updates, target_tokens=8192, bucket_size=128):
    """Plan update boundaries BEFORE microbatching; independent of model/method.

    Each epoch shuffles IDs, sorts local buckets by length, then shuffles the
    bucket order. Examples stay intact. Exact row IDs and both token counts
    are persisted. Pipelines consume contiguous slices of this same stream.
    """
    batches = []
    epoch = 0
    current, inputs, supervised = [], 0, 0
    while len(batches) < updates:
        rng = random.Random(seed + epoch)
        order = list(range(len(rows)))
        rng.shuffle(order)
        buckets = [sorted(order[i:i + bucket_size], key=lambda j: (rows[j]["input_tokens"], j))
                   for i in range(0, len(order), bucket_size)]
        rng.shuffle(buckets)
        for bucket in buckets:
            for i in bucket:
                current.append(i)
                inputs += rows[i]["input_tokens"]
                supervised += rows[i]["supervised_tokens"]
                if inputs >= target_tokens:
                    batches.append({"row_indices": current, "input_tokens": inputs,
                                    "supervised_tokens": supervised})
                    current, inputs, supervised = [], 0, 0
                    if len(batches) == updates:
                        return {"seed": seed, "target_input_tokens": target_tokens, "updates": batches,
                                "input_tokens": sum(b["input_tokens"] for b in batches),
                                "supervised_tokens": sum(b["supervised_tokens"] for b in batches)}
        epoch += 1


def microbatches(indices, rows, max_examples=2, padded_token_cap=1800):
    batch, length = [], 0
    for i in indices:
        new_length = max(length, rows[i]["input_tokens"])
        if batch and (len(batch) == max_examples or new_length * (len(batch) + 1) > padded_token_cap):
            yield batch
            batch, length = [], 0
        batch.append(i)
        length = max(length, rows[i]["input_tokens"])
    if batch:
        yield batch


def collate(rows, pad_token_id, device="cpu"):
    length = max(len(r["input_ids"]) for r in rows)
    return {k: torch.tensor(v, device=device) for k, v in {
        "input_ids": [r["input_ids"] + [pad_token_id] * (length - len(r["input_ids"])) for r in rows],
        "labels": [r["labels"] + [-100] * (length - len(r["labels"])) for r in rows],
        "attention_mask": [[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in rows],
    }.items()}


def read_jsonl(path):
    # splitlines() also splits U+2028 in legitimate mathematical prose.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line]
