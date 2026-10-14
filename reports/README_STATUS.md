# Status policy

Raw run directories are the source of truth. HPO records are immutable and are selected by endpoint `dev_tune` NLL only. Superseded runs remain on disk with a written reason and are excluded from selection. `artifacts/coverage.json` is regenerated from `artifacts/run_plan.json` and run roots; absent artifacts remain `pending`, never `complete`.
