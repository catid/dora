# Interpreting the recorded COGS length flags

The frozen round-two runner records `generation_cap_hits` by checking whether
tokenizer EOS ID 151645 appears. The pinned Qwen generation configuration also
stops on ID 151643. A response ending on that alternative EOS can therefore be
flagged despite stopping before the 640-token limit, particularly when it is
the longest response in its batch. Treat the recorded counts as
**tokenizer-EOS-absence flags with possible false positives**, not exact counts
of exhausted generation budgets.

The original raw generated token IDs were not retained, so an exact correction
is not guaranteed from decoded text. Atom exact match, strict exact match and
atom F1 are computed from saved text and do not depend on these flags.

After all 18 final COGS runs completed, the live runner was updated for future
runs to use the [helper](generation_diagnostics.py). It flags exhaustion only
when the continuation reaches the limit and none of the configured EOS IDs
appears. It handles integer, sequence and absent EOS configurations. The runner
reads `model.generation_config.eos_token_id`, leaving the `generate()` call and
all model/scoring behavior unchanged, and captures the helper source alongside
the runner.

Each future prediction includes a `generation_diagnostics` object with version
`eos_and_length_v1`, `continuation_token_ids_with_padding`, the configured
`eos_token_id`, the passed `pad_token_id`, and `max_new_tokens`. The token list
excludes the prompt and retains batch padding. EOS-terminated rows retain their
stop token, so padded length cannot falsely imply exhaustion for this
EOS-stopped protocol. The saved IDs allow both stopping tokens and actual
length to be audited without decoding ambiguity.

This is a post-run diagnostic-only fix. The 18 completed runs, their saved
predictions and metrics, and all captured executed sources were preserved
unchanged. Their legacy counts keep the limitation described above; the new
diagnostic does not retrospectively replace them.
