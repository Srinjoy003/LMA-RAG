# Step 7: the agent demonstration and SQL verification

Nilkanta Karak, 13 November 2026, on Souradeep's Step 6 (`59ba81a`). Everything was checked offline: a scripted agent, the stub grader and the stub simulator. No live model was called and no result is claimed.

## 1. The tool-calling agent, runnable

The agent driver (`src/engine/agent.py`) already existed. An LLM runs the interview by calling `search_question_bank`, `ask_candidate`, `grade_answer`, `ask_follow_up`, `update_scorecard` and `end_interview`, and code refuses any call that breaks the rules. Step 7 makes it runnable and observable:

```bash
python scripts/agent_demo.py                                    # offline: scripted agent, stubs, no key
python scripts/agent_demo.py --probe-rules                      # plus deliberate rule-breaking calls
python scripts/agent_demo.py --system A2EV --candidate CAND-002 --probe-rules --out demo.json
python scripts/agent_demo.py --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b
                                                                # live: the grader provider's model is the agent
```

The demo prints:

- every tool call in order: accepted, refused or invalid, with the reason and the milliseconds it took;
- the totals: steps, calls, refusals and invalid calls;
- the decision and levels computed by code, the agent's summary, and a summary-consistency check;
- with `--compare-code` (on by default), the same interview through the code driver: same questions, same decision, same levels, and the agent's extra model calls and wall time.

`--out` saves the full record as JSON and never overwrites an existing file.

**Offline agent.** `ScriptedAgent` (`src/engine/demo.py`) reads the last successful tool result and makes the next step of the procedure. With `--probe-rules` it first makes one deliberate rule-breaking call before several steps. Each must be refused while the interview continues unchanged:

| Rule-breaking call | Refusal |
| --- | --- |
| `end_interview` before any question | the stop rule is not met |
| `ask_candidate` with an id the rules did not choose | the rules chose another question |
| `update_scorecard` before grading | nothing has been graded |
| `verify_sql_result` on a non-SQL question | not a SQL read-code question |
| `search_question_bank` while a question is open | the question is still open |
| `grade_answer` before verifying a SQL read-code question | needs `verify_sql_result` first |
| a tool that does not exist | invalid call |

Tests check that a probed run asks the same questions and reaches the same decision as an obedient one, and that an obedient agent gives the same interview as the code driver, with and without verification.

**Summary check** (`summary_check`). This is a heuristic, not a proof. It fails a summary that:

- states a verdict the computed decision does not support, for example "strong fit" for needs-review;
- describes a needs-review outcome without calling it inconclusive or for review;
- says "<skill> level N" with a level other than the reported one.

The scripted agent's summary repeats the computed outcome, so it passes by construction. With a live model the check is real, and a failure is worth reading.

**Cost.** Each agent step is one interviewer-model call, on top of the grader and diagnoser calls the code driver also makes. Offline, the scripted agent costs nothing: a 14-exchange interview took 59 steps. The saved result records `agent.steps`, `agent.llm_seconds`, `agent.tool_seconds` and per-call `agent.tool_timings`. Measure the live overhead on a few pilot candidates before deciding to use the agent driver in an experiment. The experiments themselves still use the code driver.

## 2. `verify_sql_result(question_id)`: SQL verification

**What it does.** For an SQL read-code question ("what does this query return?"), it runs the question's own query on the question's own tables. It uses the existing runner in `src/part1/execute.py`: in-memory SQLite with a time limit. The grader then receives the true output in a "VERIFIED EXECUTION RESULT" section, so it can check, for example, that the candidate said 4 rows when the query returns 12.

**What it never does.**

- It never executes candidate text. Only bank questions are run, looked up as the question the rules chose. Running candidate code would need a separate isolation design, which is out of scope.
- It never shows the result to the candidate. The candidate's prompts are byte-identical with the tool on and off, and `state.pending` is never modified; the grader gets a copy. A test checks this.
- It never gives the agent the rows: the agent sees only the status and row count.

**Statuses.**

| Status | Meaning | Effect |
| --- | --- | --- |
| `verified` | the rows equal the bank's checked `expected_output` | rows given to the grader |
| `mismatch` | the query ran but disagrees with the bank | nothing given; report flagged (the bank needs a look) |
| `error` | SQLite error or time limit | nothing given; report flagged |

All 18 SQL read-code questions in the bank verify.

**Optional, and a separate condition.** `InterviewConfig.sql_verification` is off in every existing preset. The tool-on condition is the new preset **`A2EV`** (= A2E + verification). With the flag off, the grader's prompt, the agent's tools and system prompt, and every tool result are exactly as before, so the declared conditions and their LLM caches are unchanged (tests check this). On stubs, A2EV asks, grades and decides exactly as A2E, because the stub grader ignores the evidence.

| Driver | Behaviour with `sql_verification` on |
| --- | --- |
| Code | verifies automatically before the provisional grade of every SQL read-code question |
| Agent | gets a seventh tool and one more sentence in its prompt; `ask_candidate` says `verification_required`; `grade_answer` is refused until `verify_sql_result` succeeds |

**Saved results.**

- Every turn now records `question_type`.
- A verified turn has `verification` (status, rows, row count, `matches_bank_output`, milliseconds). This is internal audit data for the reviewer.
- The report flags any mismatch or error.

**Evaluation.** Declared as comparison **C11**, `A2-rev+verify` (A2EV) against A2-rev (A2E), on a separately identified paired subset (run `rv1-sqlverify`, chosen at the freeze). It stays outside the main matrix unless the team adds it then.

`evaluate.py` reports:

| Metric | Meaning |
| --- | --- |
| `sql_read_code_grading_accuracy` | the planned-mark check, restricted to SQL read-code questions |
| `verified_questions` | SQL read-code questions verified |
| `verification_failures` | mismatches and errors |

Results saved before Step 7 have no `question_type`. There the question is recognised from its saved prompt (read-code questions show `Table ...`; this matches all 18 and none of the 27 explain questions).

Verification runs locally, so it adds **no LLM calls**; the grader's prompt is a little longer on these questions. On stubs both conditions grade perfectly, so a difference can only come from a live grader.

```bash
python scripts/run_experiment.py --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b \
    --candidates-dir data/candidates_validation/pilot --systems A2E A2EV --run rv1-sqlverify --operator nilkanta
python scripts/evaluate.py --run rv1-sqlverify --method A2EV --baseline A2E --comparisons
```

## Files

| File | Change |
| --- | --- |
| `src/engine/verification.py` | new: the tool, statuses, grader and agent views |
| `src/engine/tools.py` | `verify_sql_result` with its rules; the grader gets the verified copy |
| `src/engine/session.py` | code driver verifies before grading |
| `src/engine/agent.py` | seventh tool and prompt sentence only when on; grading refused until verified; per-call timing |
| `src/engine/result.py` | turns record `question_type` and `verification` |
| `src/part3/grader.py` | "VERIFIED EXECUTION RESULT" section, only when present |
| `src/shared/interfaces.py` | `sql_verification` flag (off) and the `A2EV` preset |
| `src/engine/demo.py`, `scripts/agent_demo.py` | new: scripted agent, rule probes, summary check, driver comparison |
| `src/part4/metrics.py`, `scripts/evaluate.py` | verification metrics |
| `docs/experiment_matrix.json`, `docs/evaluation_protocol.md` | `A2-rev+verify`, C11, `rv1-sqlverify` |
| `tests/test_step7_tools.py` | 21 tests; `tests/test_engine_agent.py` and `tests/test_experiment_matrix.py` updated |

Any Step 7 source change alters the manifest fingerprint, so runs made before it cannot be resumed. Use new run names.

## Handoff to Souradeep

1. **Review the interaction with grading.** The grader gets the verified rows only for SQL read-code questions with status `verified`. The diagnoser gets nothing. Check that this does not change the diagnosis or evidence behaviour you built in Step 4, and that the grader prompt section is phrased as you want.
2. **Confirm the core conditions are unchanged.** With the flag off, prompts, tools and tool results are identical. The only changes to existing results are two added turn fields (`question_type`, and `verification` when on) and the agent's timing fields.
3. **Step 8 (pilot), if budget allows:**
   - a few live agent-driver interviews, to measure its real overhead and check the summary;
   - A2E against A2EV on the pilot candidates, to see whether verification changes SQL read-code grading.
