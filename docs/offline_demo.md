# Offline demonstration: agent rules and SQL verification

Owner for preparation/presentation: Souradeep. This builds on Nilkanta's existing Step 7 implementation; it requires no API credentials and makes no network calls. It demonstrates execution and enforcement, not live model quality.

## Run (about a second locally, excluding Python startup/environment setup)

From the repository root, using an environment with the project's runtime dependencies:

```bash
python3 scripts/agent_demo.py --grader stub --simulator stub --system A2E \
  --candidates-dir data/candidates_validation/pilot --candidate VAL-P-004 \
  --probe-rules --out results/demo/agent-off.json

python3 scripts/agent_demo.py --grader stub --simulator stub --system A2EV \
  --candidates-dir data/candidates_validation/pilot --candidate VAL-P-004 \
  --probe-rules --out results/demo/agent-on.json
```

Outputs are not overwritten. For a repeat, choose new output filenames. Explicit `stub` arguments ensure that locally configured credentials are not used. No final evaluation candidates are needed.

## Present in this order (three minutes)

1. Explain the separation: the scripted agent calls tools, code enforces interview rules, and stub tags supply grades/diagnoses. This is a reproducible demonstration, not a live LLM interview.
2. Show the refused early `end_interview`, wrong-question `ask_candidate`, premature scorecard update and search while a question is open. Show the invalid unknown tool separately.
3. In A2EV, show `grade_answer` refused before SQL verification. Then show SQL-L2-014 verified with 3 rows. Only the bank query/tables run; candidate SQL never runs. The grader receives rows; the agent gets status/counts.
4. Show the computed recommendation and levels, then the heuristic summary check. Both examples recommend human review, with Python 3, SQL 1 and data structures 1. This is a synthetic example, not a hiring conclusion.
5. Show each agent/code comparison: matching questions, levels and decisions, 14 exchanges each. A2E has 56 scripted steps and A2EV has 61 with the refusal probes. These are not billable LLM calls or live latency measurements.

The example with SQL verification performs three bank checks: SQL-L2-014, SQL-L2-010 and SQL-L1-008. All verify; both conditions retain the same candidate prompts and outcomes on stubs. A real grader's response to execution evidence remains unmeasured.

## Saved examples and troubleshooting

[demo/offline_summary.json](demo/offline_summary.json) records the checked outputs and source fingerprint; [demo/agent_sql_trace.txt](demo/agent_sql_trace.txt) is the captured verification-on console trace. Full generated JSON stays under ignored `results/` when you run the commands above. Millisecond timings vary; compare outcomes and refusal reasons rather than exact timestamps.

- If `--out` exists, use another filename; do not delete experiment manifests to rerun a demo.
- If imports fail, use the configured project environment. These CLI demos do not need Streamlit, model downloads or a live server.
- Summary consistency is a heuristic and the scripted agent passes by construction. Do not describe it as demonstrated live summarisation reliability.
- Leave the code driver as the experiment driver unless an explicitly separate agent experiment has been declared and budgeted.
