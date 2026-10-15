# Generation/parser preflight (v3)

The greedy generation runner was exercised on 16 public GSM8K TRAIN examples
with a 64-token smoke limit. It produced zero generation errors and zero parser
timeouts. Fourteen generations reached the smoke limit without EOS; the parser
classified 15 outputs as unsupported, which is expected for an unadapted base
model under a deliberately short smoke limit. Public TRAIN reference parsing is
covered separately by `tests/test_v3_answer_parser.py` (512 references, at
least 500 supported numeric answers).

No `dev_audit`, GSM8K test, or MATH-500 test answers were opened by this
preflight. Official evaluation remains sealed until selection/configuration
freeze.
