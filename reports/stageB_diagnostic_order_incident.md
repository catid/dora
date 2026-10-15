# Stage B diagnostic-order incident

The 492-trial HPO completed successfully with no failed cells. A long-lived Stage B supervisor process had been started before the final declared-diagnostics transition code was loaded. When the HPO barrier cleared, that process proceeded directly to full-horizon tuning before launching the four registered deletion diagnostics.

The supervisor process (PID 3446058) was stopped before screening could begin. The four tuning jobs already running were left untouched. After tuning completes and releases the GPUs, the diagnostics for `L_GATE` and `I_GATE` at seeds `2901` and `2902` will run sequentially against the exact replay checkpoints. Screening will remain held until those four summaries exist.

This incident is selection-ineligible and preserved in [the machine-readable receipt](../artifacts/protocol_incidents/stageB_diagnostic_order_20261015.json). HPO and tuning measurements are not being treated as final quality results while the incident is open.
