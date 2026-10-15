# Mechanism diagnostics (v3)

These CPU toy probes are shadow diagnostics and are not used to choose a method.
They test signed first-order deletion estimates, a finite-difference curvature
proxy, interacting deletions, and gauge rescaling. The actual-update expression
is restored after each probe; the adapter digest matches before and after for
both initializations.

| Init | Seed | Gauge forward change | Pair interaction (slots 0,1) | Restoration |
|---|---:|---:|---:|---|
| L | 2903 | 0.000000 | 0.000001 | pass |
| I | 2904 | 0.000000 | 0.010999 | pass |

The centered I probe shows a measurable pair interaction, so independent slot
first-order scores cannot be treated as exact joint deletion costs. The L probe
interaction is effectively zero for this draw. The curvature and signed
first-order values are preserved in `artifacts/mechanism_diagnostics_v3.json`.
