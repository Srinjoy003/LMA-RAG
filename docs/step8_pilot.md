# Step 8: stopped live pilot record

Owner: Souradeep. Status on 14 November 2026: **stopped; no live process or automatic continuation**. Kushal owns any subsequent readiness and final execution. See [implementation_handoff.md](implementation_handoff.md) for the current guide. This document preserves the attempted pilot, not an instruction to restart it.

## Declared plan and actual scope

The pre-run plan was all 12 frozen VAL-P candidates across B1/A2/A2E/A4/ADAPT12 (60 interviews). A2 was included to compare the suspected-corroboration accuracy trade-off directly. No final CAND candidates were run. A2EB, A2EV and live agent checks were optional and not launched. The original 48-interview budget excluded A2; neither estimate guaranteed a practical wall time.

Souradeep chose Groq for both roles: Qwen `qwen/qwen3.8-27b` grading/diagnosis/topic extraction and GPT-OSS `openai/gpt-oss-120b` simulation, both with low reasoning effort. Both credentials authenticated, and both models passed small strict-JSON compatibility calls. Small compatibility calls did not establish full-interview reliability. The primary key served both roles; the spare was not assumed to provide additional quota. Credentials remain local and are not part of this record.

## What happened

| Run | Outcome |
| --- | --- |
| `rv3-groq-pilot-20261114` | Two B1/VAL-P-001 attempts failed on HTTP 400 `json_validate_failed`; no completed interview |
| `rv4-groq-pilot-20261114` | One B1/VAL-P-001 completed; A2E/VAL-P-001 failed because all diagnosis votes were unusable; full batch never started |

B1 asked 12 main questions, stopped at fixed length and had no interview errors. Elapsed time was **11 minutes 24 seconds**, including waits, simulation and cache reuse. It recorded 37 successful interviewer completions, 27 uncached, with 35,680 input and 15,672 output tokens across successful calls. These counts include locally cached completions where applicable, omit failed HTTP attempts and simulator usage, and are not an invoice or a representative final-run runtime estimate. One candidate supplies no reliable accuracy comparison.

The background driver stopped at its A2E smoke guard. Souradeep then rejected the runtime of the large pilot; no further calls are scheduled. Original results, manifests, cache and failures are preserved locally under ignored `results/`. Do not include credentials or raw provider diagnostic payloads in a shared handoff.

## Reviewed client change

Groq's exact HTTP 400 `json_validate_failed` code is now classified as `LLMOutputError`, consistent with schema-invalid HTTP-200 outputs. The existing grader/diagnoser can drop the unusable vote. Ordinary invalid-request errors remain fatal. Rejected generations are never accepted, repaired into answers or cached, and no automatic retry was added. All-invalid vote batches still fail, as the A2E smoke demonstrates. The initial failure captured max_tokens=2048 and a malformed extra field; more tokens were not established as a remedy.

The source change required a new rv4 manifest. Neither old attempts nor outputs are relabelled as successful new interviews. Valid cached responses may be reused by the existing cache policy. Base engine is `18d8be6` plus the recorded client patch; a final committed handoff must include that patch.

## Offline audit and remaining evidence

`docs/step8_pilot_plan.json` is an archived plan/status record. `scripts/audit_pilot.py` checks completeness and provenance against it, restricts the population to VAL-P, reports wall times and stop reasons, and adds the direct A2E/A2 comparison when both exist. It makes no model calls and must report the single-result run as incomplete:

```bash
python3 scripts/audit_pilot.py --run rv4-groq-pilot-20261114 \
  --json results/rv4-groq-pilot-20261114/stopped_pilot_audit.json
```

Exit code 2 means incomplete/flagged; it is expected here. Output files are not overwritten. This audit is not a final CAND-run auditor. Its unit tests cover missing pairs, duplicate pairs, unexpected final candidates and midnight-spanning runtime.

The client fix and supporting regression tests are reviewed offline. Qwen evidence-mode compatibility remains unresolved. Human-supported precision and human-human kappa are not measured. No revised-system gain, stopping calibration, tool improvement or completed live pilot is claimed. The next empirical checks belong to the final executor's separately agreed model/budget setup.
