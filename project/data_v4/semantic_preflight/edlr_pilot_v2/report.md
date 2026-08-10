# EDLR pilot v2 offline preflight

- Development rows: 30 (10 per source).
- Arms: EA_TOP5, UNION_LLM, EDLR, EDLR_SHUFFLE.
- Prepared completion payloads: 120.
- Network/API calls: **0**; cost: **0**; every payload is `NOT_SENT`.
- Candidate-union size: min 7, mean 9.43, max 13.
- B0 reproduction, target-free construction, outer-training-only evidence, union identity, shuffle non-identity, vocabulary, and literal leakage gates: PASS.

A new explicit authorization is required before `/models` or any of the 120 billed completions.
