# Recorded fit costs

Collected 5562 physical fits from 3 hash-verified reports. No model inference or metric changes were performed.

`costs.json` retains every source snapshot, raw timing field and report view; `fits.csv` has one row per physical fit. `raw_costs.csv` contains the literal timed observations with field paths and hashes. Raw rows are not additive.

Physical paths deduplicate fixed/tuned views, transfer datasets, repeated input reports, and retrieval seed-42 pilot/final training. Distinct pilot and final retrieval evaluations remain separate, including each corpus encoding. The final result overwrites the pilot evaluation, so both hashed snapshots are necessary. Initial evaluations copied between snapshots count once.

Retrieval evaluation totals include corpus encoding plus query/ranking split times. Never add those totals to their children or count the NFCorpus-trained SciFact transfer view again. Aircraft training includes epoch validation; Qwen training includes periodic validation. Aircraft test timing and Qwen generation timing are separate. Qwen test-NLL time is unavailable.

Initialization uses only the recorded injection timer. Nested preprocessing/decomposition/prefit timings can describe historical cache creation, appear under many modules, or overlap that timer. They are retained as observations, never added as new fit cost. Missing values stay null/blank; a recorded zero stays zero. No total is fabricated, and timestamp intervals are not used because they can span scheduling gaps. Controls, audits, shared calibration, reloads, failed partial work without timers and device warmup are outside this accounting. Failed fits are retained with their available measurements.

Every input report output and evidence file was verified against its recorded SHA256. Receipt-linked result/initialization files are additionally checked against the receipt hashes. This verifies provenance, not the completeness of historical timers.
