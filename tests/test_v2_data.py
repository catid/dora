import json

from rankstudy.v2.data import (QuestionOverlap, connected_groups, effective_manifest,
                              microbatches, read_jsonl, stratified_take)


def test_connected_aliases_group_augmented_problems():
    rows = [{"original_question": "problem 1", "query": "paraphrase"},
            {"original_question": "paraphrase", "query": "reverse"},
            {"original_question": "problem 2", "query": "independent"}]
    assert connected_groups(rows) == [[0, 1], [2]]
    assert rows[1]["query"] == "reverse"


def test_decontamination_preserves_numerals_and_handles_near_duplicates():
    q = "Alice has 12 apples and Bob has 4 apples in a small red basket today. How many apples do they have in total?"
    overlap = QuestionOverlap([q])
    assert overlap.match(q.upper()) == "normalized_exact"
    assert overlap.match("Please answer: " + q) == "near_trigram_jaccard"
    assert overlap.match(q.replace("12", "13")) is None


def test_effective_batch_ids_and_token_counts_independent_of_microbatch():
    rows = [{"input_tokens": 200 + i * 17, "supervised_tokens": 100 + i} for i in range(40)]
    plan = effective_manifest(rows, 1001, updates=12)
    for update in plan["updates"]:
        for micro in (1, 2, 4):
            actual = [i for b in microbatches(update["row_indices"], rows, micro) for i in b]
            assert actual == update["row_indices"]
        assert update["input_tokens"] >= 8192
        assert update["input_tokens"] == sum(rows[i]["input_tokens"] for i in update["row_indices"])
    assert plan == effective_manifest(rows, 1001, updates=12)


def test_example_sampling_does_not_require_one_example_per_group():
    chosen, remaining = stratified_take(list(range(30)), 20, 4, lambda i: str(i % 3))
    assert len(chosen) == 20 and not set(chosen) & set(remaining)


def test_unicode_prose_is_not_a_record_boundary(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"text": "first\u2028second"}, ensure_ascii=False) + "\n")
    assert read_jsonl(p) == [{"text": "first\u2028second"}]


def test_group_and_token_weighted_metrics_are_distinct_when_lengths_differ():
    # This mirrors the metric definition used by the v2 trainer: each group
    # gets one vote in the first metric, each supervised token in the second.
    losses = [(1.0, 10), (3.0, 100)]
    group = sum(x for x, _ in losses) / len(losses)
    weighted = sum(x * n for x, n in losses) / sum(n for _, n in losses)
    assert group == 2.0
    assert weighted != group
