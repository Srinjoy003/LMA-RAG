# Step 4: quoted diagnostic evidence and disputed findings

Implemented as the new experimental **A2E** preset. Human difficulty ratings and answer annotation remain deferred; the checks here use synthetic cases and offline stubs.

## What changed

| Preset | Confirmation | Diagnostic evidence |
| --- | --- | --- |
| A2 | Two replies | Historical prompt and behaviour |
| A2Q | Two distinct main-question IDs | Historical prompt and behaviour |
| A2E | Two distinct main-question IDs | Evidence prompt, exact-quote checks, strict voting, contradiction tracking |
| A4 | One reply | Historical reduced diagnosis package |

Historical presets and default experiment conditions are preserved. Explicitly select A2E for this increment. Its configuration saves `diagnostic_evidence: true` and derived `diagnosis_prompt_version: evidence-v1`; the latter is metadata emitted by `to_dict`, not an extra constructor argument. Actual diagnostic replies also carry the prompt version. Changed prompts and schemas are part of the LLM cache key.

```bash
# One offline interview, no key needed:
python scripts/run_interview.py --system A2E --job data-analyst

# Offline pipeline comparison, use a new folder:
python scripts/run_experiment.py --systems A2 A2Q A2E A4 --run step4-evidence-stub
```

These examples are plumbing checks, not live evaluation. The engine, factory, application selector, and agent driver can all use the new preset. Streamlit's default remains A2; select A2E explicitly. Existing CLI experiment choices are derived from the system list.

## Supporting evidence

Evidence mode requests an exact substring of the **current candidate reply**, together with a reason explaining how it expresses the specific listed misconception. The prompt distinguishes merely wrong answers from the particular wrong belief: O(n) heap removal is not evidence for a listed O(1) misconception.

Each vote is cleaned before aggregation. A missing, blank, fabricated, or reference-only quote, missing reason, or conflicting support/contradiction cannot supply positive evidence. Unsupported misconception votes become `gap` and do not trigger a targeted follow-up for that unsupported claim. A positive finding requires agreement from strictly more than half of the configured votes, not just a plurality or one usable vote out of three. The engine independently rechecks quote provenance for custom diagnosers before recording support.

Each recorded supporting evidence entry retains the existing question ID, turn, reply index, reply text, and vote share, and adds:

- `supporting_quote`: verbatim candidate text;
- `quote_start`, `quote_end`: zero-based Python character offsets, end exclusive; the first occurrence is used for repeated text;
- `reason`: the diagnoser's explanation of the match.

The substring check verifies provenance. It cannot prove that the model interpreted the statement correctly, understood negation, or chose a truthful explanation. No human-validated diagnostic precision is claimed.

## Contradictions and status

The evidence prompt can identify explicit rejection or correction of a listed belief. Silence, omission, a different misconception, and unrelated correct statements are not intended as contradictions. Contradictory claims need their own verbatim quotes and reasons, and a strict majority of configured votes. Repeating the same contradictory claim several times within one vote counts as only one vote.

When a belief was previously attributed to the candidate, a valid contradiction is appended to its `contradicting` list using the same provenance fields plus `reason`. The finding becomes `status: disputed` and `confirmed: false`, even if its positive question count remains two or greater. Positive evidence and the original chronology are retained for audit.

Statuses are `suspected`, `confirmed`, and `disputed`. A rejection of a belief never previously attributed to the candidate does not create a new misconception finding; the diagnosis still retains the contradiction. Reports and the interviewer table display disputed status explicitly. The existing candidate prompt rendering does not receive diagnostic evidence.

**Conservative policy:** a disputed finding cannot automatically become confirmed again during that interview. Later positive votes do not erase a contradiction. It remains available for further probing, subject to existing question/follow-up limits. This is an explicit unresolved-conflict state, not a claim of permanent belief or an implementation of belief recovery. Any future recovery rule needs a separate version and validation.

## Correct replies and call budget

Historical systems skip diagnosis when the provisional grade is 2. A2E diagnoses correct main replies too, allowing an explicit correction to challenge a previously attributed belief. A `correct` diagnostic cause is available in evidence mode. A provisionally correct main answer does not receive a follow-up. Correct follow-up replies can also end the current exchange.

This adds diagnosis calls on correct main answers. The historical average of 104 calls and Nilkanta's pre-Step-4 budget estimates must not be reused as measurements of A2E. Measure calls, length, precision, recall, and deferral on a separate pilot before freezing the experiment. The grader, scorecard, and stopping threshold were not changed in this increment.

## Schema and compatibility

New evidence/status/contradiction fields are optional in saved results. The diagnosis cause enum also accepts `correct`; historical result examples still validate. `confirmed` remains the compatibility boolean: disputed findings are false. Positive `confirmation_count` is an audit count, not the final status when contradictory evidence exists.

The summary and interviewer table show the finding's status. Detailed quotations and contradictions are available in saved JSON; a separate evidence-browser UI has not been added. Legacy metrics can still count confirmed findings, but disputed counts must be reported explicitly by Nilkanta's evaluation extension.

Offline TagDiagnoser supplies simulator-tag text as quote evidence, labelled `tag-evidence-v1`. This exercises wiring; it does not perform semantic diagnosis or demonstrate live-model quality. The adversarial and correction cases in tests use explicitly synthetic replies.

## Tests and verification

Focused coverage includes quote offsets with Unicode, absent/fabricated/reference quotes, strict voting and partial failures, duplicate contradiction votes, conflicting support, engine-side rejection, correct-answer contradiction detection, disputed-state persistence, report snapshot isolation, new-preset factory wiring, both drivers, and historical result compatibility.

Run:

```bash
python -m pytest
python scripts/validate_bank.py
```

No API calls are required for these checks. Live accuracy and the benefit of this change remain unmeasured.

## Handoff to Nilkanta

1. Review A2E as the proposed successor to A2Q for `A2-rev`; keep both older presets for named controls.
2. Amend the experiment matrix/protocol explicitly before freezing it. Add the evidence flag to expected configuration differences and create the job-blind counterpart of the chosen revised preset.
3. Mark deferred human-supported endpoints as not measured. Do not replace them with model-generated annotations described as human labels.
4. Extend metrics with suspected/disputed counts, quote-provenance rejection counts, and cost; avoid treating a disputed finding as confirmed because its positive count is high.
5. Include evidence/prompt source hashes in the safe-run manifest. Use new run paths and freeze candidate/data versions.
6. Re-estimate the pilot and final run budget, since A2E diagnoses correct main replies as well.

Step 6 stopping calibration, additional tool integration, final experiments, and report result updates remain separate tasks. No automatic commit or push is performed by this implementation.
