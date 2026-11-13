# Handoff review record — 14 November 2026

Owner: Souradeep. This is a code/document consistency review and offline verification record, not independent human annotation or a live-model acceptance certificate.

| Area checked | Finding / action | Status |
| --- | --- | --- |
| Client JSON failure handling | Exact HTTP-400 `json_validate_failed` becomes an output error; other invalid requests remain fatal; no rejected payload accepted/cached | Reviewed and tested |
| Incomplete vote handling | Valid grading votes survive one unusable output; all-invalid grading and evidence-diagnosis batches still fail | Regression tests passed; model reliability not established |
| Full regression suite | 596 tests passed, offline | Passed |
| Bank correctness checks | 135 questions, 95 misconceptions, 3 jobs; all 54 read-code checks executed | 0 errors, 0 warnings |
| Agent demo without SQL verification | VAL-P-004, 14 exchanges, 56 probed scripted steps; questions/levels/decision match code driver | Passed, no network calls |
| Agent demo with SQL verification | Same candidate, 14 exchanges, 61 probed steps; three SQL questions verify, candidate prompts unchanged | Passed, no network calls |
| Summary consistency | Both scripted summaries agree with computed outcomes | Passed by scripted construction; heuristic, not live validation |
| Demo source provenance | Saved summary's code hash matches current `src/**/*.py` | Checked |
| Simulator report description | Removed false claim that it never sees model answers; documented correct/slip/misread reference access | Corrected |
| A4 and memory claims | Described bundled changes; deferred A2M cannot establish a memory-only effect | Corrected |
| Deferred annotation | Removed promises of imminent hand-labelling; human-supported precision and kappa remain unmeasured | Corrected |
| Premature-stop interpretation | Retained historical breakdown; removed inference of equal per-skill rates from counts alone | Corrected |
| Revised methods | Added distinct-question evidence, disputed state, level-band uncertainty, stopping alternative and SQL tool, with limits | Written in `report/implementation_update.tex` |
| Historical results | Numeric macros and historical figures preserved, with old-bank scope and missing raw transcripts stated | Checked |
| Current live status | One B1 smoke completed, A2E smoke failed, full batch never started, no automatic continuation | Archived as incomplete |
| Pilot completeness audit | Actual saved run gives 1/60 saved, 59 missing and exit 2 | Correct rejection of completion |
| Final experiment instructions | Main presets, explicit populations, model resolution, freeze/resume/failure rules and pairing documented | Prepared; no final execution performed |
| Figure reproducibility | Existing figure generator selects B1/A2, not A2E; invalid revised-matrix command removed | Limitation documented; revised plots pending outputs |
| PDF build | Tectonic generated 14-page report; new sections rendered and inspected; no undefined references or overfull text | Passed; only existing float-placement warnings |
| Links and credentials | New guide links resolve; reviewed text contains no configured API keys or Groq-key literals | Checked |
| Protected materials | Original README, mid_report.pdf, bank/candidates and historical numbers/figures not changed; assessment stays outside repository | Preserved |

## Response to the four guide/TA remarks

1. **Premature stops:** keep the original historical per-skill/level breakdown, distinguish counts from rates, describe the boundary-aware alternative and its inconclusive stub trade-off. Historical raw interviews are not present, so this review does not claim a fresh recomputation of that experiment.
2. **A4 priority:** A2E/A4 is central to RQ2 in the revised guide/report. It tests the combined diagnosis/candidate-belief package, not persistent memory in isolation.
3. **Low kappa:** checks were deferred. The report preserves a conditional plan to investigate ambiguity, revise guidance, adjudicate and collect fresh labels if reinstated. The chosen 0.6 criterion is not a measured result or universal guarantee.
4. **Theory of mind:** the report explains the concrete belief tracker, evidence/probe behaviour and the comparison that evaluates it, without claiming a validated psychological model.

## What remains genuinely dependent on new evidence

- Kushal's chosen-model readiness check, resource budget, final freeze and execution.
- Final metric tables, revised-system plots/macros and conclusions using the resulting versioned outputs. `docs/final_results_template.md` is intentionally unfilled; the historical figure generator must not be used unchanged for A2E.
- Optional live stopping/tool/agent comparisons, only if selected. Additional independent human validation and A2M remain outside the present scope.

No additional model calls, commit or push were made while completing this consolidation. Original authorship is retained; Souradeep now owns the remaining review, writing and preparation work.
