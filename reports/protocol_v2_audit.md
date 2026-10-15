# Protocol audit, 2026-10-15

The 166 terminal v2 HPO records are preserved at their original paths. They are
**not eligible for primary method selection**. This audit supersedes earlier
statements that the v2 reference arm was faithful and that 959 runs represented
the complete requested scope. Data v2's intact question/response pairs and sealed
streams remain in use; this is a training/protocol correction, not new sampling.

| Confirmed discrepancy | Evidence in v2 | Consequence |
|---|---|---|
| Custom score EMA updates only at allocation events | `run_one_v2.py`, custom score loop inside `%16` condition | Different estimator/time constant from every-update EMA |
| AdaLoRA smooths aggregated slots | `adalora_scores()` returns per-slot means before EMA/uncertainty | Not upstream element-level smoothing; reported AdaLoRA quality cannot support a faithful-reference claim |
| AdaLoRA final mask reapplied only every 16 steps | Reference block in `run_one_v2.py` | Dormant E coordinates can regrow during the final hold interval, unlike upstream's every-step final masking |
| Slot budget counts triplet amplitudes in the *target* | Target expression adds the triplet `+1` | Exceeds the common two-factor scalar upper bound |
| Progress measured in steps | `(step+1)/args.updates` | Does not implement exact token-progress allocation |
| Freeze boundary not forced | `%16` or end-of-training allocation only | No explicit allocation at first crossing of 80% tokens |
| Factor storage contradicts documentation | `A,B` parameters; `U,V` are subtraction properties | Not actual trainable displacement storage; small changes may be lost |
| Every method collects dense G and input activations | Unconditional hooks in `StudyLinear.forward` | Artificial overhead in simpler baselines; timing comparison invalid |
| Triplet/reference forward materializes dense updates | Calls `delta_expanded()` in branch | Avoidable full-matrix work, not the declared low-rank arithmetic |
| No physical compaction or resumable checkpoint | Native runner writes metrics and metadata only | No memory reduction, exact resume, actual-update export, or checkpoint deletion diagnostics |
| Provenance revision recorded at run *end* | `git rev-parse HEAD` after training | Insufficient proof of code executed at run start |
| Plan omits required runs | Only one shortlisted config per tuning seed; D/E have three baseline cells; rank4 excludes DISP/ACT | 959 is not the full requested study denominator |

The v2 supervisor, continuation shell, and four verified child workers were
stopped. Exact commands/PIDs are recorded in
`artifacts/protocol_v2_stop_receipt.json`. No unrelated jobs were stopped and no
measurements were deleted. Quarantine records include original metadata hashes.
Interrupted work is an engineering failure, not an algorithmic loss.

The replacement is isolated in `src/rankstudy/v3` and `scripts/run_one_v3.py`.
The original AdaLoRA adapter/allocator source and license are vendored verbatim
from revision `d10f5ebee16c478fa2f41a44a237b38e8c9b0338`; dependency packages are
not vendored. A trajectory test compares parameters, element EMAs, and final
masking to the upstream implementation through 40 actual optimizer updates.
The new custom branches store real displacements, use low-rank forward/gate/ACT
algebra, update task scores every effective batch before clipping, clear pruned
moments before stepping, and compact survivor parameters and Adam state at the
token freeze boundary. Checkpoint tests verify exact subsequent optimizer/RNG
behavior. Passing these tests alone does not establish complete study compliance.

Pending gates include complete LLM smoke coverage, full-horizon schedule/resume
checks, DDP parity, pipeline audits, generation/parsers, deletion diagnostics,
all required repetitions, and statistical analysis. No new primary winner is
declared. Dev-audit and official test outputs remain sealed.
