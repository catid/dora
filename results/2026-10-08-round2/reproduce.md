The archive contains scored predictions, task/source manifests, training logs, local audit attestations, compact teacher evidence when available, and this report’s source. It excludes downstream model and adapter checkpoints. All included files have SHA-256 hashes. Publication uses deterministic 4 MiB parts named raw_artifacts.tar.gz.part-000 onward; raw_manifest.json and summary.json list their ordered names, byte counts and SHA-256 hashes, plus the complete archive hash. The final part can be smaller. Download every listed part, raw_manifest.json and report_source before rebuilding.

From this report directory, reassemble and check every part and the complete archive using only the Python standard library. The assembly command refuses missing, reordered, truncated or hash-mismatched parts and writes the archive atomically after successful verification. Then rebuild the downstream tables and PNG/SVG figures without a GPU, model downloads, or original absolute paths:

```bash
python report_source/experiments/second_round/archive.py --assemble-parts --report-dir .
mkdir raw
tar -xzf raw_artifacts.tar.gz -C raw
PYTHONPATH="$PWD/raw/report_source" CUDA_VISIBLE_DEVICES='' python -m experiments.second_round.report \
  --raw-root "$PWD/raw" --output-dir "$PWD/rebuilt" --skip-bundle
```

The report verifies every archive-manifest hash before scoring and re-scores saved classification predictions, retrieval rankings and COGS text. It independently recomputes both Aircraft initialization controls and the separate GPU-norm recalibration probe from retained logit arrays. It also regenerates the matched-factor-LR magnitude diagnostic from validation-only trial files; that diagnostic does not use test scores or change the primary results. It validates available checkpoints, and explicitly counts omitted payloads. Full original checkpoint audits remain local audit attestations; omitted downstream weights cannot be revalidated from this compact archive. Rebuilding figures is not rerunning model inference or training.

To repeat the packaged offline reconstruction audit from this report directory:

```bash
PYTHONPATH="$PWD/raw/report_source" CUDA_VISIBLE_DEVICES='' python -m experiments.second_round.verify_report_rebuild \
  --report-dir "$PWD" --work-parent /tmp --force-parts --audit-output "$PWD/offline_rebuild_audit_reproduced.json"
```

This reconstructs the archive exclusively from its published parts, ignoring any locally retained complete archive, then extracts a fresh read-only evidence tree, runs outside the repository with offline model-library settings, and compares generated figures, tables, diagnostics, sources and structured task scores. It verifies every input and part hash again afterward. The separate audit records part/complete-archive hashes, output/source hashes and environment; it stays outside that archive to avoid circular hashing. The verifier also uses parts automatically when the complete archive is absent.

Optional full task-level portable checks:

```bash
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.analyze_vision raw/vision/rank2_run --skip-checkpoints
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.analyze_vision raw/vision/rank8_run --skip-checkpoints
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.analyze_cogs --root raw/cogs --allow-missing-checkpoints
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.validate_retrieval raw/retrieval
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.validate_control_vision raw/vision/numerical_control --no-write
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.validate_control_vision raw/vision/numerical_control_full --no-write
```

Run the report before optional analysis scripts: those scripts can rewrite analysis files, changing archive-manifest hashes. The report also runs teacher_compact/verify.py: it regenerates all synthetic problem tensors exactly, verifies retained small adapter checkpoints, and independently reconstructs dense test predictions on CPU. Teacher scores and plots remain separate from downstream quality scores. The full report needs Python, NumPy, Matplotlib and CPU PyTorch because it runs the compact teacher verifier automatically. No GPU or model download is required; use the recorded package versions for exact reproduction. Original absolute paths in provenance describe original runs and do not locate files during this rebuild.

COGS diagnostic audit scope: The executed COGS runner recorded `hit_generation_cap`/`generation_cap_hits` by checking whether tokenizer EOS 151645 was absent. The pinned generation configuration also stops on EOS 151643, so these flags can falsely indicate token-limit exhaustion. Raw generated token IDs were not saved, preventing an exact retrospective cap audit. Saved text and its atom-set exact match, strict exact match, validity and F1 scores are unaffected. The archive retains the executed source; later EOS-detection or raw-token-logging fixes apply only to future runs and do not change these records.
