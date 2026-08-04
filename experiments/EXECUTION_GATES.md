# Execution Order and Human Decision Gates

Status: LOCKED

```text
T0.1 -> T0.2 -> GATE_1 -> T0.3
  -> E1 -> GATE_2
  -> E2 -> E3 -> GATE_3
  -> E4 -> E5 -> E6
  -> E7 -> E8 -> E9
  -> PHASE_4_REPAIR
```

## Mandatory gate policy

1. GATE_1 question: Have all four discrepancies been located and supported by an evidence chain?
   - Current state: `WAITING_FOR_HUMAN`
   - If not located: stop.
   - Only an explicit human confirmation may authorize T0.3.
2. GATE_2 question: Is the semantic branch effective under the predeclared E1 criterion?
   - Current state: `WAITING_FOR_HUMAN`
   - If ineffective: stop.
   - Only an explicit human confirmation may authorize E2.
3. GATE_3 question: Does semantic fusion survive the semantic-versus-pure-ensemble control?
   - Current state: `WAITING_FOR_HUMAN`
   - If the predeclared death condition is met: stop and report `DEAD`.
   - Only an explicit human confirmation may authorize E4.

The Agent may prepare evidence and recommend a decision, but may not approve a gate. Silence, an Agent conclusion, or completion of the preceding task does not count as human approval.

One experiment is run at a time. After its artifacts and decision report are complete, execution stops for human review.
