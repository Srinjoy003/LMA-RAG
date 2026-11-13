# Final results worksheet — unmeasured until outputs are supplied

Prepared by Souradeep. This is a writing template, not a result. Replace **pending** only from the frozen run's saved outputs. Do not import the historical B1/A2 numbers or the stopped smoke into this table.

## Provenance and completeness

Run: **pending**. Code commit: **pending**. Fingerprint: **pending**. Bank/catalog hashes: **pending**. Models, providers and effort: **pending**. Scope and dates: **pending**. Readiness outcome and changes since readiness: **pending**.

| Condition | Saved unique candidates | Error-flagged | Excluded after retries | Missing | Paired denominator |
| --- | --- | --- | --- | --- | --- |
| B1 | pending | pending | pending | pending | per comparison |
| A2E | pending | pending | pending | pending | per comparison |
| A4 | pending | pending | pending | pending | per comparison |
| ADAPT12 | pending | pending | pending | pending | per comparison |
| A5E | pending | pending | pending | pending | per comparison |

Explain every missing/excluded pair without excluding based on outcomes. Resolve provenance warnings before using outputs. State which optional conditions were actually run.

## Results to populate

| Metric | B1 | A2E | A4 | ADAPT12 | A5E |
| --- | --- | --- | --- | --- | --- |
| Skill accuracy (`skill_accuracy`) | pending | pending | pending | pending | pending |
| MAE (`mae`) | pending | pending | pending | pending | pending |
| Mean main questions / exchanges | pending | pending | pending | pending | pending |
| Confirmed recall / precision | pending | pending | pending | pending | pending |
| False alarms on correct replies | pending | pending | pending | pending | pending |
| False fit / false rejection / deferral | pending | pending | pending | pending | pending |
| Confident / cap / exhaustion stops | pending | pending | pending | pending | pending |
| Successful / uncached interviewer completions | pending | pending | pending | pending | pending |
| Whole-interview wall time | pending | pending | pending | pending | pending |

B1/ADAPT12 have no diagnostic findings; undefined precision is not zero precision. Distinguish proportions from percentages and percentage-point differences. Include paired candidate-bootstrap 95% intervals (2,000 draws, seed 0) for declared comparisons. State undefined bootstrap draws and do not replace them with zero.

### RQ1

Report A2E minus B1 skill-accuracy difference and exchange difference, both intervals and denominators. Assess the declared non-inferiority/exchange target using the protocol, even if not met. Report ADAPT12 versus B1 separately as the equal-exchange adaptive selection package. Do not explain a longer interview's advantage as adaptivity alone.

### RQ2

Report A2E versus A4 recall, precision and false alarms together with length and cost. State A4's bundled differences. Summarise suspected/disputed counts, quote rejections and selected transcript errors, without calling the quote check semantic validation. No isolated persistent-memory claim without a matched memory control.

### Error and resource analysis

For premature stops, show wrong/assessed skills by skill and true level, over/under-estimates, errors among true-level-0 skills, all-stop errors and cap rate. For fit decisions include decided-case accuracy with its denominator and the deferral fraction. Review usable grading/diagnosis vote counts, simulator-plan deviations, manipulation examples and the effect of weakness-focused questions on skill estimates.

Separate successful interviewer completions, uncached completions, simulator work, cache reuse, HTTP retries/failures and observed wall time. State exactly which totals are unavailable. No cost extrapolation from unmetered stubs or a single smoke.

### Optional tools/stopping

Only populate if actually run: stopping paired comparisons and cap/all-stop trade-off; A2EV/A2E SQL read-code planned-mark alignment, verified question counts, failures and length; live agent summary consistency, refusals and overhead. A scripted demo is implementation evidence, not a live result.

## Limitations and conclusion

Human-supported precision: **not measured**. Human-human kappa: **not measured**. Real-candidate validity: **not established**. The simulator sometimes sees reference answers, and planted truth is not an adjudication of written replies. Final CAND profiles were frozen but previously inspected in stub analyses. Any missing or unstable provider runs remain a limitation.

Answer each RQ from the intervals and trade-offs, including negative/inconclusive findings. Keep old-bank historical numbers in their historical section. State whether each proposed revision was adopted, rejected or left unmeasured, and why. Remaining work is not a completed contribution or measured result.
