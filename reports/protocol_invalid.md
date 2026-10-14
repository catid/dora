# Superseded native HPO records

The first native screen (`/var/tmp/dora-adaptive-rank/runs/hpo_native2`, 348 runs) used one two-example microbatch per optimizer update and is excluded because it did not meet the declared effective-update token exposure.

The second screen (`/var/tmp/dora-adaptive-rank/runs/hpo_corrected`, partial) accumulated intact examples to 8192 input tokens and normalized by supervised tokens, but it was stopped after the faithful AdaLoRA orthogonality audit and before completion; its partial rows are retained and excluded.

The third screen (`/var/tmp/dora-adaptive-rank/runs/hpo_final`, partial) used those corrections but preceded the LR schedule audit; its partial rows are retained and excluded.

The final accepted screen is `/var/tmp/dora-adaptive-rank/runs/hpo_accepted_final2`, with global supervised-token accumulation, padding-aware ACT sampling, faithful AdaLoRA orthogonality, monotonic custom masks, reference post-step AdaLoRA allocation, microbatch-1 memory policy, and 3% warmup/cosine decay to 10% peak. The launcher OOMs from abandoned attempts were caused by GPU-slot bookkeeping/orphan workers and are retained as launcher failures, not algorithm results.
