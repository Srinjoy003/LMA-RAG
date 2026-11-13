# Step 6: stopping uncertainty and review of Step 5

Souradeep's implementation, 13 November 2026, on Nilkanta's Step 5 (`8722b2b`). All checks below used offline stubs. No live-model experiment was started. Step 3 remains deferred; no human validation is claimed.

## Decisions after review

- Accept the CRLF-to-LF hashing policy. It normalises checkout line endings, not JSON contents or candidate semantics. The existing frozen hashes agree after this normalisation. Keep refusing incompatible results instead of silently resuming them.
- Keep `ADAPT12`, `A5E`, `A4`, `A2`, `A2Q` and `A2E` separately identifiable. Keep `A2E` as the declared revised system until a pilot supports a change.
- **Defer A2M.** It is optional, adds another condition and a tracker-reset design, and is unnecessary for the combined diagnosis-package comparison. Without it, do not claim the experiment isolates persistent memory. No approval from another teammate is needed to continue Steps 6/7 with that scope.
- Add a separately named stopping alternative and the missing uncertainty fields. Do not change the score model, level cuts, candidate truth, or the weight of incorrect corroboration replies to improve the reported accuracy.

## Step 5 control and manifest audit

| Condition | Review outcome |
| --- | --- |
| `ADAPT12` | Adaptive question choice; fixed 12 main questions; diagnosis and generic follow-ups off. Thus exactly 12 exchanges, including all follow-ups (there are none). It matches the earlier figure-script control. It retains A2's weak-topic/probing ranking, so describe the B1 comparison as the adaptive selection package, not an isolation of SE-based skill selection alone. |
| `A5E` | Only `retrieval` differs from A2E. Job-blind retrieval removes the job-topic preference, while diagnosis and corroboration remain enabled. |
| `A4` | Historical settings preserved. It bundles diagnosis vote count, follow-up behaviour and confirmation changes; it is not a matched-memory ablation. |
| `A2` / `A2Q` / `A2E` | Remain distinct: historical reply confirmation, distinct-question confirmation, and evidence-backed diagnosis respectively. |

The manifest hashes **all `src/**/*.py`**, including prompt code outside its displayed `PROMPT_SOURCES` list. The six existing displayed sources were covered. I expanded that list to also display topic extraction (`src/part1/extract_topics.py`), the agent system prompt (`src/engine/agent.py`) and follow-up wording (`src/engine/followups.py`). These were already protected by the full source hash; this improves visibility.

The Step 6 source/configuration changes and expanded manifest display require a **new run folder**. Never edit an old manifest to resume it. Windows and macOS users must run the same source revision and configurations; the manifest still detects genuine differences after line-ending normalisation. Tests cover control differences and refusal to mix changed setups.

## Investigation of the accuracy drop

I reproduced Nilkanta's existing 60-candidate **stub** observation as a descriptive audit, not a threshold search or a new final/live result:

| Condition | Skill accuracy | Mean exchanges | Confirmed planted-misconception recall |
| --- | ---: | ---: | ---: |
| A2 | 78.3% | 15.30 | 56.1% |
| A2Q | 67.8% | 15.87 | 40.2% |
| A2E | 67.8% | 15.87 | 40.2% |

Across A2Q/A2E there were 33 main questions selected specifically to corroborate a suspected misconception; every one scored zero in the stub simulation. The simulator forces a holder's relevant misconception answer to fail, while true skill levels are the floored average of topic levels without a misconception penalty. Repeatedly sampling those weaknesses can therefore lower the scorecard estimate relative to that particular ground truth. A2Q and A2E agree here because tag-based stub evidence always validates; these data cannot establish the effect of real quote validation.

As a temporary diagnostic only, suppressing the suspected set used by selection in A2E recovered accuracy to 76.7%, but recall fell to 7.3% (16.12 exchanges). That intervention was not added as a preset, accepted as a fix, or used to change the final conditions. It shows why removing corroboration priority is not a free improvement. Stopping confidence cannot resolve a mismatch between a one-dimensional score model and uneven abilities/misconceptions.

The original evaluation set was already inspected for earlier stub analyses. Only the separate `VAL-P` population below was used to check the new stopping alternative; the final live evaluation remains pending. No threshold sweep was performed.

## Implementation and field contract

Every new final `report.levels[skill]` and `InterviewSession.scorecard_view()` contains:

| Field | Meaning |
| --- | --- |
| `level_probability` | Posterior mass inside the band of the **reported EAP level** |
| `level_probabilities` | Mass in all four bands, with JSON keys `"0"` through `"3"` |

Bands use the existing cuts `-1.25`, `-0.25`, `0.75`: level 0 is below -1.25, level 1 is [-1.25, -0.25), level 2 is [-0.25, 0.75), level 3 is at least 0.75. Probabilities use the existing finite grid [-4, 4] and sum to one before serialization. They describe uncertainty **conditional on the score model**, not measured probabilities that the candidate's ground-truth label is correct. The scalar is not the maximum of the four masses: an EAP level can differ from the most probable band.

Both fields are optional in the saved-result schema, so old files remain readable. Values are rounded to six decimals for display/storage; stopping uses full precision. The interviewer scorecard and final table display `P(reported level)`, with blanks for old results. The UI confidence label now covers either stopping rule rather than always calling it an SE-only check.

| Preset | Reference | Behavioural difference |
| --- | --- | --- |
| `A2EB` | A2E | `stop_rule="level_confidence"` |
| `A2B` | A2 | Same stopping alternative, for an optional matched historical comparison |

The new rule requires **all three skills** to have at least two graded main questions, SE strictly below 0.6 and mass at least 0.7 in their reported level. It retains the 15-main-question cap, bank exhaustion handling, question selection and follow-ups. Thresholds are configurable; invalid mass thresholds fail before the interview. Exhaustion without enough confidence is reported as `bank_exhausted`, never `confident`. A cap stop remains `max_questions`, even if some individual skills meet the rule.

The 0.7 gate was chosen before running the separate stub check as a single experimental setting. It is not calibrated or accepted as an improvement. Existing presets still use their original SE-only stopping behaviour. Fit/review decisions retain their separate existing job-minimum probability rules; level uncertainty is not a new verdict rule. The agent uses the same stopping safeguards as the code/session driver.

## Separate offline validation

Run `rv2-step6-check`: 12 frozen pilot candidates (`VAL-P-001`–`012`, seed 3102), five conditions, **60 completed interviews, zero errors**. This checks method behaviour and saved-result plumbing; it is not the paid/live pilot of Step 8. Selected metrics and provenance are in [step6_stub_pilot.json](step6_stub_pilot.json); complete transcripts, configurations and the manifest are generated under ignored `results/`.

| Preset | Correct skill levels / 36 | Exchanges | Confident stops / 12 | Cap stops / 12 | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| A2 | 30 (83.3%) | 15.50 | 12 | 0 | 20% |
| A2Q | 30 (83.3%) | 15.58 | 12 | 0 | 15% |
| A2E | 30 (83.3%) | 15.58 | 12 | 0 | 15% |
| A2B | 32 (88.9%) | 17.17 | 0 | 12 | 20% |
| A2EB | 31 (86.1%) | 17.67 | 0 | 12 | 15% |

**Interpretation:** A2EB adds 2.08 exchanges for one more correct skill level. Its paired accuracy difference versus A2E is +2.8 percentage points, with a candidate-bootstrap 95% interval of -5.6 to +11.1 points (2,000 draws, seed 0). It does not establish an accuracy improvement, and it disables early confidence stops on this small stub population. Keep it experimental rather than promoting it to the main method.

- True-level-0 error is **1/2 assessed skills for every condition**, an overestimate; this has not been solved. The denominator is too small for a reliable rate claim.
- Each condition makes zero false-fit and zero false-reject decisions out of 12 interviews, with **7/12 deferred**. This is not perfect decision accuracy on all candidates.
- Confirmed recall stays 15% for A2E/A2EB (3/20 planted misconceptions); it does not improve with the longer interview. Stub precision is 100% because tags directly reveal the diagnosis; it is not evidence of live precision.
- The confident-and-wrong interview rate falls from 6/12 to 0/12 for A2EB **because no interviews stop confidently**. Its five wrong skill levels are still included in overall accuracy. Report cap rate and all-stop errors alongside that metric.
- Stub LLM/API calls are zero. The extra exchanges and diagnosing correct answers can raise live cost. Nilkanta's estimates remain estimates; budgets for the original conditions exclude the extra stopping comparisons.

## Reproduce

From the repository root, using the project's Python environment:

```bash
python scripts/run_experiment.py --grader stub --simulator stub \
  --systems A2 A2Q A2E A2B A2EB \
  --candidates-dir data/candidates_validation/pilot \
  --run rv2-step6-check --operator souradeep

python scripts/check_stopping.py --run rv2-step6-check \
  --json results/rv2-step6-check/stopping_audit.json

python -m pytest -q
```

The audit requires a versioned, error-free `VAL-P` run and refuses final `CAND` results. It reports paired comparisons, all-stop errors by true level, low-band-mass errors, stop reasons, and confident-only errors with denominators. It refuses to overwrite its JSON output: choose another output filename when repeating the analysis. Use another run name after source/configuration changes. The matrix declares C9/C10 for the stopping comparisons without changing the main conditions.

Regression coverage includes exact band boundaries, posterior normalisation, EAP versus modal bands, insufficient evidence, invalid thresholds, cap/exhaustion stops, unchanged controls, code/session/agent execution, and old saved-result schema compatibility. The full suite passes **566 tests**.

## Handoff to Nilkanta and next steps

1. **Step 7:** implement the runnable agent demonstration and optional trusted-bank SQL verification. Keep the declared comparison conditions and grader unchanged unless a separately named tool-on condition is used. A2M and human annotation are not prerequisites.
2. Audit the new fields and C9/C10 definitions before accepting any stopping variant. Retain all-stop level errors and fit/review rates; a lower confident-stop error rate alone is insufficient.
3. **Step 8:** compare A2E/A2EB on the same separate live pilot candidates if the extra cost fits the agreed scope. Adding A2EB adds 12 interviews to the original 48-interview pilot, before any optional A2/A2B diagnostics. Measure calls, latency, cap rate and actual transcript behaviour. A2EB is not the default and the present stub evidence supports keeping A2E as the reference.
4. Choose settings and update the actual full/job-blind preset pair, matrix, budgets and run version together at the Step 9 freeze. If the alternative does not improve the trade-off, retain SE-only stopping and report the rejected alternative honestly. No new final experiments or live performance claims have been made here.
