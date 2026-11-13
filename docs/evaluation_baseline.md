# Reviewed-bank baseline and Souradeep's first handoff

Recorded 13 November 2026. Starting revision: `910981fefcb07e0145ee5c9d6daff6daf30c4165`.

The machine-readable [baseline manifest](evaluation_baseline.json) records SHA-256 hashes of all bank inputs and all 60 existing candidate files. This is a starting data snapshot, not a claim that the new implementation is already committed at that revision. Record the eventual implementation commit with each experiment.

## Inputs and availability

| Input | State |
| --- | --- |
| Question bank | 135 checked questions; unchanged by this work |
| Misconception catalog | 95 entries; unchanged by this work |
| Candidate keys | Existing 60 files retained, individually hashed |
| Historical experiment | Report describes 24 candidates × B1/A2, on the pre-review bank |
| Original `results/midsub/` | Unavailable locally; shared location unknown |
| External review export | Unavailable locally; corrected bank itself is present |
| Original live configurations | Not independently verified without the result JSONs |

Missing historical results prevent original-transcript annotation and independent reproduction of old live figures. They do not prevent development, new validation runs, or preparation of the evaluation harness. New data must not be labelled as the old experiment. Do not regenerate the candidate profiles silently.

## Implemented increment

`A2Q` is a separately named experimental version of A2. It changes the confirmation unit from replies to distinct main-question IDs. The original `A2`, `A1`, `A3`, `A4`, and `A5` presets retain their reply-counting policy; A4 still confirms from one reply.

```python
from shared.interfaces import SYSTEMS

historical = SYSTEMS["A2"]
revised = SYSTEMS["A2Q"]
# Equivalent policy override for a separately named custom comparison:
custom = historical.with_(system="my-question-control",
                          misconception_confirmation_unit="question")
```

The field `misconception_confirmations` remains the threshold; `misconception_confirmation_unit` selects `reply` or `question`. Both are saved in result configuration. In question mode, a question ID is required. Several follow-up replies, or the same question ID recorded on another turn, contribute only one confirmation. All diagnosed replies remain in the audit evidence.

The new preset is available through the existing system selector and CLI choices:

```bash
python scripts/run_interview.py --system A2Q --job data-analyst
python scripts/run_experiment.py --systems A2 A2Q A4 --run question-confirmation-stub
```

These commands use offline stubs. An experiment with live grader/simulator providers is a separate, budgeted validation task. The experiment runner's default system list is unchanged, so request `A2Q` explicitly. Use a new run folder; old results are skipped by filename rather than checked against the manifest.

## Evidence contract for Nilkanta

Each newly produced finding retains `id`, `confirmed`, `evidence_turns`, and the mean diagnosis `vote_share`. It also includes:

| Field | Meaning |
| --- | --- |
| `confirmation_unit` | `reply` or `question` |
| `confirmation_count` | Number of units satisfying the current counting rule |
| `required_confirmations` | Configured threshold |
| `evidence` | Records for all replies on which the misconception won the diagnosis |

Each evidence record stores `question_id`, `turn`, `reply_index`, `reply_text`, and `vote_share`. Index 0 identifies the main answer; indices 1 and 2 identify follow-ups under that main question. `reply_text` is the exact submitted answer, not a generated supporting quotation. Older saved results need not contain any of the new fields and remain schema-valid. Consumers should handle their absence.

The existing mean vote share remains an average across replies, including related follow-ups. It is an audit statistic, not a calibrated belief probability or an independence-weighted score.

**Limits:** this increment fixes the counting rule and records provenance. It does not change the LLM diagnosis prompt, require semantic support for a label, extract verified supporting quotes, resolve contradictions, calibrate stopping, or establish improved precision. Distinct IDs provide operational corroboration, not proof of statistical independence. Contradiction handling and semantic evidence validation remain later diagnosis work after independent annotation.

Stricter confirmation can cause additional follow-ups and can reduce recall. The main-question cap and per-question follow-up limit still apply. The result summary now describes an unconfirmed finding as having insufficient corroboration rather than assuming it was seen only once.

## Verification

Executed against this working increment:

```bash
python -m pytest -q -p no:cacheprovider
python scripts/validate_bank.py
```

- **473 tests passed**, including 12 new policy/provenance cases.
- Bank validator: 135 questions, 95 misconceptions, all 54 read-code questions executed; **zero errors and warnings**.
- New tests exercise repeated evidence from one question, reusing the same ID on another turn, corroboration across distinct questions, A4/historical preservation, exact reply provenance, snapshots, invalid policy settings, and both code and agent drivers.
- Existing example-result schema checks still pass.
- No live model calls or new evaluation claims were made.

## Next handoff

Nilkanta should review the evidence contract and prepare the annotation sheet, experiment matrix, and versioned runner/reporting support. Treat `A2Q` as a confirmation-policy comparison against `A2`; it does not isolate all aspects of the candidate-belief model. Keep A4 separately identified. Freeze candidate files and data hashes across conditions.

Both reviewers still need independent annotations. Record unknown historical provenance instead of filling it with guesses. The historical log request can proceed in parallel.

Suggested files for Souradeep to stage after reviewing the diff:

```bash
git add src/shared/interfaces.py src/engine/selection.py src/engine/followups.py src/engine/report.py schemas/interview_result.schema.json tests/test_confirmation_policy.py docs/evaluation_baseline.md docs/evaluation_baseline.json
git commit -m "Add distinct-question confirmation and reviewed-bank baseline"
git push
```

Only the named implementation and handoff files should be staged; local PDFs and `.DS_Store` files are unrelated. This handoff document does not itself commit or push the work.
