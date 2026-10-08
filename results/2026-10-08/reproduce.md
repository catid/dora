The raw archive contains metrics, predictions, split manifests, exact executed source snapshots, and verification logs. It excludes model weights and full-resolution images.

From the repository root, with the experiment Python environment installed:

```bash
restored_results=/tmp/dora-results-restored
mkdir -p "$restored_results"
tar -xzf results/2026-10-08/raw_artifacts.tar.gz -C "$restored_results"
python -m experiments.report \
  --vision-root "$restored_results/vision" \
  --retrieval-root "$restored_results/retrieval" \
  --extraction-root "$restored_results/extraction" \
  --sdxl-root "$restored_results/sdxl" \
  --sdxl-extra-roots "$restored_results/sdxl_seed43" "$restored_results/sdxl_seed44" \
  --verification-root "$restored_results/verification" \
  --output-dir "$restored_results/rebuilt-report" --skip-images --skip-bundle
```

This recomputes the reported metrics from archived predictions and generates the tables and PNG/SVG chart without a GPU, model downloads, or original absolute paths. It does not rerun inference. The committed SDXL montage remains available separately; regenerating it requires the original full-resolution PNGs. Absolute paths inside provenance describe the original run and are not needed for this metric/chart reproduction command.

When the original full-resolution generated PNGs are available, rerun the separate CPU class-only CLIP diagnostic with:

```bash
python -m experiments.sdxl_clip_diagnostic --run-dirs \
  /var/tmp/dora-bench/sdxl /var/tmp/dora-bench/sdxl_seed43 /var/tmp/dora-bench/sdxl_seed44
```

This writes all three class_clip_metrics.json files. It changes scoring text only, leaving generations, model selection, and original CLIP scores unchanged. It requires the generated PNGs and the pinned CLIP checkpoint; it cannot run from the compact metric archive alone.
