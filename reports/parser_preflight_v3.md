# Evaluation parser preflight

Math-Verify is a candidate reference, not yet installed or selected. The inspected source at ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b uses SymPy `parse_expr` in the expression extraction path. `evaluate=False` is not a security boundary and does not satisfy the instruction never to evaluate model strings as Python.

Next: audit a restricted LaTeX-only path and its backend, or build expressions from a safe grammar and use the reference symbolic equivalence logic on those objects. Parser validation must use public TRAIN examples, subprocess timeouts and a frozen extraction policy before test unsealing.

Source/provenance: `../artifacts/parser_preflight_v3.json`. No dependency change or test-set access occurred.
