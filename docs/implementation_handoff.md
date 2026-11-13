# Implementation handoff and experiment guide

Prepared by Souradeep, 14 November 2026. Souradeep owns this consolidation, review, writing and demo preparation. Existing implementation authorship is preserved. Kushal owns subsequent model/budget selection, readiness runs and final execution. This document does not start any API calls or transfer credentials.

## Current decision

Implementation and offline checks are ready for review. Live revised-system performance is **not validated**. Souradeep's large pilot is stopped and must not resume automatically. One B1 smoke completed, A2E failed, and the full 60-interview batch never started. Keep A2E as the declared reference, A2EB/A2B experimental, A2EV optional, A2M deferred and human annotation unperformed. No model accuracy/precision/length improvement follows from passing unit tests.

Start with [offline_demo.md](offline_demo.md) for a three-minute demonstration, [step8_pilot.md](step8_pilot.md) for the stopped run, and `report/implementation_update.tex` for the report-ready methods/limitations. The original README and historical numerical macros are preserved.

## What code is being handed over

| Work | Main source | Evidence / original revision |
| --- | --- | --- |
| Distinct-question confirmation and reply provenance | `src/engine/followups.py`, `src/shared/interfaces.py` | Souradeep, `c297b3f`; A2Q versus historical A2 |
| Difficulty-packet preparation | `scripts/prepare_difficulty.py` | Souradeep, `a922df4`; preparation only, no completed ratings |
| Quotes, corroboration, contradictions and disputed findings | `src/shared/evidence.py`, `src/part3/diagnoser.py`, tracker/report/UI | Souradeep, `b0cfda0`; A2E and `docs/step4_evidence.md` |
| Controls, manifests, pairing, failure handling | `src/part4/`, evaluation scripts | Nilkanta, `8722b2b`; `docs/step5_evaluation.md` |
| Level-band uncertainty and experimental stopping | `src/engine/scorecard.py`, selection/report/session | Souradeep, `59ba81a`; A2EB/A2B and `docs/step6_stopping.md` |
| Agent demonstration and trusted SQL verification | `src/engine/verification.py`, agent/tools, `scripts/agent_demo.py` | Nilkanta, `18d8be6`; A2EV and `docs/step7_tools.md` |
| Groq output-error classification | `src/llm/client.py`, `tests/test_llm_client.py` | Included in this handoff commit; freeze its committed revision before final execution |
| Stopped-pilot audit and consolidation | `scripts/audit_pilot.py`, Step 8 docs, report, demo examples and this guide | Current Souradeep-owned work; no further work assigned to Nilkanta |

The client patch handles only HTTP 400 with code `json_validate_failed` as an unusable output. It does not repair/accept `failed_generation`, retry it automatically, weaken the evidence majority, or turn all-invalid votes into a diagnosis. Grading can retain fewer valid votes, which affects agreement weights. The A2E smoke still failed after this patch; it is **not a demonstrated Qwen compatibility fix**. Ordinary invalid-request errors remain fatal.

## Conditions and claims

| Question | Conditions | What the comparison permits |
| --- | --- | --- |
| RQ1 accuracy/length | A2E vs B1 | Compare skill accuracy and exchanges jointly; include fit errors and deferrals |
| RQ1 equal length | ADAPT12 vs B1 | Adaptive selection package at exactly 12 exchanges, including weak-topic ranking |
| RQ2, primary diagnostic comparison | A2E vs A4 | Combined diagnosis/candidate-belief package, not memory alone |
| Job-aware retrieval | A5E vs A2E | Only retrieval differs |
| Optional historical-rule check | A2E vs A2; A2Q vs A2 | Separate evidence/confirmation changes using named presets |
| Optional stopping | A2EB vs A2E; A2B vs A2 | Stop-rule change only; compare cap rates and all-stop errors |
| Optional SQL verification | A2EV vs A2E | Verification-on/off paired subset, including SQL read-code grading alignment |

Recommended main matrix remains **B1/A2E/A4/ADAPT12/A5E**, 60 frozen candidates each, 300 interviews. This is a proposed scope, not an obligation or a runtime estimate. Freeze the chosen scope before execution. Do not silently substitute A2 for A2E or A5 for A5E. If A2EB is adopted later, create its matched job-blind counterpart before claiming a retrieval-only comparison. An agent-driver run is separate from the code-driver matrix.

## Manual execution sequence for Kushal

All commands below are instructions for the executor; none were launched by this handoff. Replace provider/model placeholders with the executor's agreed settings. Keep the same grader/simulator setup across comparison conditions, use different model families, and check actual model access. No assumptions about today's provider offerings or quota are made here.

1. Pull the commit containing this handoff and the client patch. Record the commit and clean source/data status. Do not change source or prompts while a run is active. A dirty-source manifest fingerprints the actual files but is harder to distribute; freeze a committed revision before final execution.
2. Configure local credentials. Do not copy another person's `.env` or assume two keys add quota. Read model names from saved `turn.grade.grader` and `simulator.model`; the historical config may contain `finetuned-grader`, which the factory resolves to its real prompted-model fallback.
3. Run offline checks, bank validation and the demo. Use the existing candidate files; do not regenerate the final CAND population.

```bash
python3 -m pytest -q
python3 scripts/validate_bank.py
# Demo commands and expected output: docs/offline_demo.md
```

4. Use a **fresh readiness folder** and validation candidates only. First check B1/VAL-P-001; inspect the saved result before starting A2E/VAL-P-001. These two interviews are a bounded engineering check, not a balanced accuracy study or a completed pilot. Interview caps bound questions, not wall time; quotas can make even one interview slow. No automatic 60-interview continuation is recommended.

```bash
# Replace GRADER_PROVIDER, SIM_PROVIDER and SIM_MODEL. Use a fresh folder after any setup change.
python3 scripts/run_experiment.py --grader GRADER_PROVIDER --simulator SIM_PROVIDER \
  --simulator-model SIM_MODEL --candidates-dir data/candidates_validation/pilot \
  --ids VAL-P-001 --systems B1 --run reviewed-readiness-v1 --operator kushal

python3 scripts/run_experiment.py --grader GRADER_PROVIDER --simulator SIM_PROVIDER \
  --simulator-model SIM_MODEL --candidates-dir data/candidates_validation/pilot \
  --ids VAL-P-001 --systems A2E --run reviewed-readiness-v1 --operator kushal
```

Inspect errors, final/provisional vote counts, grade/diagnosis content, quote validity, question coverage, stop reason, resolved models and usage. An all-invalid diagnosis is a failed check, not evidence that the candidate has a knowledge gap. A2E has not passed this check on Souradeep's all-Groq setup. Changes after readiness need a new version and their own check. If broader calibration is needed, agree a bounded validation scope and budget first; do not restart the abandoned large pilot by default.

5. Freeze code, bank/catalog and candidate hashes, seeds, models/providers/effort, prompts, conditions, main versus optional scope, failure rules and budget. Record the readiness outcome and outstanding limitations. Do not call one successful B1 smoke a full-system readiness pass.
6. Run the final matrix only after readiness and freeze. Explicit CAND ids keep validation data out of the final run. These are the existing 60 final profiles (previously inspected in stub analyses, not a pristine unseen stub holdout).

```bash
python3 scripts/run_experiment.py --grader GRADER_PROVIDER --simulator SIM_PROVIDER \
  --simulator-model SIM_MODEL --candidates-dir data/candidates \
  --systems B1 A2E A4 ADAPT12 A5E --run reviewed-main-v1 --operator kushal

python3 scripts/evaluate.py --run reviewed-main-v1 --method A2E --comparisons \
  --json results/reviewed-main-v1/metrics.json

```

**Figure-script limitation:** `report/make_figures.py` currently selects B1/A2 explicitly. It cannot generate the recommended revised main matrix figures (which has A2E and no A2); running it on that matrix would fail. Use `evaluate.py`'s JSON as the numerical source. Final revised-system plots/macros must be generated after the outputs arrive and the selected conditions are known; do not run the historical script or relabel old A2 figures as A2E. `report/numbers.tex` and the existing result figures remain historical. This is an explicit reporting task pending actual results, not a working final-figure command.

## Resume, exclusions and audit

- The runner skips saved pairs only after the manifest setup/configuration/candidate checks. Changed code, bank, prompt, models or old unversioned folders require a new run name. Never delete/edit a manifest to force a resume.
- Run the same command to resume. Ordinary failures accumulate across invocations and are excluded after three failed attempts for that condition/candidate pair. Quota stops are resumable and do not count as failed attempts. The runner's exit code alone does not establish completion.
- Expect 60 candidates per main system, unique `(system, candidate_id)` pairs, and all declared comparisons' paired denominators. Keep excluded pairs and errors visible. A completed result with a non-empty error list needs review rather than silent removal.
- `evaluate.py` **prints provenance warnings**; it does not itself guarantee contaminated results were filtered out. Resolve every warning before reporting metrics. Check actual condition configurations, expected ids, missing systems and duplicate copies as well.
- `scripts/audit_pilot.py` is tied to the stopped VAL-P plan; it is not a final-run auditor. It deliberately labels the single-result run incomplete. Use the final matrix and manifests when auditing CAND results.
- Keep original transcripts, manifests, cache, failure logs and resolved model settings with the run. Live outputs are ignored by Git; share through an agreed results artifact, never a credentials archive. If sharing across machines, compare fingerprints before merging and preserve both session manifests as in Step 5.

## How to report the measurements

Use [final_results_template.md](final_results_template.md) after actual outputs arrive. Skill accuracy is over candidate-skill pairs; precision/recall use confirmed findings; disputed findings are not confirmed. Report uncertainty and per-comparison pairing. Missing endpoints remain undefined/not measured, not zero. Human-supported precision and kappa remain unmeasured because annotation was deferred.

Report false fits, false rejections and deferrals together. Confident-stop error needs its confident-stop denominator, cap frequency and all-stop errors; a rule that never stops confidently is not automatically better. Simulator plan alignment measures grader plus simulator behaviour, not independent grading accuracy. Topic-level truth ignores planted-misconception penalties and can disagree with weakness-focused sampling.

`mean_llm_calls` counts successful interviewer completions; `mean_api_calls` excludes local cache hits but also omits failed/retried HTTP attempts. Simulator calls are outside those totals. Token sums can include cached completions. Wall time includes throttling/waits and is not pure model latency. Do not infer invoices or final-run duration from a stub's zero cost or one cached smoke.

## Review, packaging and authorship

Preserve the original README, `mid_report.pdf`, local assessment, TA remarks, artifact-link text, `.env`, unrelated `.DS_Store` files and generated raw runs. Do not stage them with `git add .`. The review-ready files are the client/test change, pilot audit/test/plan/status docs, this guide, demo examples/guide, report source/implementation section/PDF, final-results template, and current protocol/TODO amendments. This handoff is shared as a reviewed commit; record its actual commit id from Git before freezing a run. The protected/local materials are not included.

Historical results remain attributed to the earlier experiment. Nilkanta's controls, runner and tool implementation remain attributed to their original commits; Souradeep now handles the remaining review, writing, packaging, demo preparation and subsequent result interpretation. Experiment execution alone does not replace those implementation contributions.

Ready-for-review is not final-experiment acceptance. Remaining evidence work is the chosen-model readiness check, the final execution and its analysis; optional features and human checks are not secretly marked complete.
