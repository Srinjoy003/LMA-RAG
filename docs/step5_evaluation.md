# Step 5: controls and reproducible evaluation

Nilkanta Karak, 13 November 2026. This implements protocol section 7 ([evaluation_protocol.md](evaluation_protocol.md)) on top of Souradeep's Step 4 (`b0cfda0`, [step4_evidence.md](step4_evidence.md)). Everything here was checked offline on stubs. No live run was made, so no result is claimed.

## What changed

| Area | Change | Files |
| --- | --- | --- |
| Conditions | A2-rev is now **`A2E`**. New presets: **`ADAPT12`** (equal-exchange control) and **`A5E`** (A2E with job-blind retrieval). Historical presets are unchanged. | `src/shared/interfaces.py`, `docs/experiment_matrix.json` |
| Run versioning | `results/<run>/manifest.json` with a fingerprint, the configurations, candidate hashes and sessions. Different setups are refused before any interview, and every result is tagged. | `src/part4/manifest.py`, `src/part4/runner.py` |
| Exclusions | Failed attempts go to `failures.jsonl`; a pair is excluded after 3. Quota stops are resumes. | `src/part4/runner.py`, `src/part4/manifest.py` |
| Candidates | Explicit ids, id files, the matrix allocation, and other folders. Validation ids such as `VAL-P-001`. The 60 frozen keys can no longer be overwritten by accident. | `scripts/run_experiment.py`, `scripts/make_candidates.py`, `src/part4/candidates.py`, `schemas/candidate.schema.json` |
| Pilot population | 12 candidates, seed 3102, generated and frozen | `data/candidates_validation/pilot/` |
| Metrics | Suspected and disputed counts; recall and precision on any named finding; quote rejections, contradictions and suspected-probe questions; cost; topic coverage; premature-stop rates with denominators; paired comparisons | `src/part4/metrics.py`, `scripts/evaluate.py` |
| Report | `--bank-root` and `--bank-label`; refuses a bank that does not match the run's manifest; uses the `ADAPT12` preset and the moved premature-stop functions | `report/make_figures.py` |
| Docs and tests | Protocol amended, this guide | `docs/evaluation_protocol.md`, `docs/part4.md`, `tests/test_step5_evaluation.py`, `tests/test_experiment_matrix.py` |

## Conditions for rv1

| Condition | Preset | Differs from A2 |
| --- | --- | --- |
| B1 | `B1` | fixed 12 questions, no diagnosis or follow-ups |
| A2-rev | `A2E` | distinct-question confirmation, quote-backed diagnosis, contradictions and disputed status |
| A4 | `A4` | single diagnosis, generic follow-up, single-reply confirmation (five bundled differences, protocol 1.1) |
| ADAPT12 | `ADAPT12` | adaptive choice, exactly 12 main questions, no diagnosis or follow-ups: exactly 12 exchanges |
| A5-rev | `A5E` | as A2E, plus job-blind retrieval |
| optional | `A2`, `A2Q` | for C5 (confirmation rule) and C8 (evidence requirement) |

`tests/test_experiment_matrix.py` fails if a preset drifts from these declared differences. It checks that A5E differs from A2E only in retrieval, and that ADAPT12 equals the report's original `adaptive12` control.

## Commands

**Pilot** (Step 8). Each person runs half of `VAL-P-001`–`VAL-P-012`, and both use the same run name:

```bash
python scripts/run_experiment.py --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b \
    --candidates-dir data/candidates_validation/pilot \
    --ids VAL-P-001 VAL-P-002 VAL-P-003 VAL-P-010 VAL-P-011 VAL-P-012 \
    --systems B1 A2E A4 ADAPT12 --run rv1-pilot --operator nilkanta
```

The other half is `VAL-P-004` to `VAL-P-009`. That is the same k ∈ {0, 3} / {1, 2} rule as the final split, so each half has 3 fit, 3 not fit, 3 even and 3 uneven.

**Final run** (Step 10, after the Step 9 freeze):

```bash
python scripts/run_experiment.py --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b \
    --allocation nilkanta --systems B1 A2E A4 ADAPT12 A5E --run rv1-main --operator nilkanta
```

**Evaluation:**

```bash
python scripts/evaluate.py --run rv1-pilot --method A2E --comparisons --json results/rv1-pilot/metrics.json
```

This prints:

- provenance problems, and pairs excluded after 3 failures;
- the metric table and the four targets;
- wrong levels after confident stops, with denominators;
- each declared comparison, with its paired candidates, the difference and a 95% interval. Draws where an endpoint is undefined are counted, not scored.

`human_supported_precision` is printed as *not measured*, because Step 3 was deferred.

**Report figures:**

```bash
python report/make_figures.py --run rv1-main          # bank checked against the run's manifest
python report/make_figures.py --run midsub --bank-root ../bank-673ee08 --bank-label historical
```

`--bank-root ../bank-673ee08` is a separate checkout of the bank commit the historical run used (`git worktree add ../bank-673ee08 673ee08`).

## Run versioning, in practice

- **New run:** the first session writes `manifest.json`. Later sessions with the same code, bank, prompts and providers append to it. Adding a condition or new candidates is allowed.
- **Refused before any interview:**
  - the engine source, bank, prompt or evidence source, or the grader/simulator setup differs from the manifest;
  - a condition's configuration differs;
  - a candidate key differs;
  - the folder has results but no manifest (for example the historical `midsub`).

  The message names what differs. Use a new run name (`rv2-...`); never edit or delete the manifest to get past it.
- **Fingerprint:** hashes of LF-normalised text, so a Windows checkout (`core.autocrlf=true`, which this machine uses) and a macOS checkout of the same commit agree. On this machine the raw bytes of every bank and candidate file differ from Souradeep's baseline hashes; the normalised hashes match. A test now checks all 60 frozen candidates against `evaluation_baseline.json`.
- **Commit and dirty flag:** recorded per session for information. The fingerprint uses the files themselves, so an uncommitted engine change is caught even without a commit.
- **Merging two machines' shares** (Step 10):
  1. Both people run the same commit with the same run name, and so get the same `fingerprint_id`. Check this in both manifests before merging.
  2. Copy the other person's `<system>/<job>/*.json` files into your run folder.
  3. Keep their manifest beside yours as `manifest.<operator>.json`, for their sessions.
  4. Run `evaluate.py`. It flags any result whose fingerprint is not this run's.

## Not done, and why

- **A2M** (matched-memory control) is not implemented. It changes the tracker and selection, which are Souradeep's modules, and the protocol makes it optional. Without it, the RQ2 claim stays limited to the combined diagnosis package.
- **Human checks** (Step 3) were skipped. The annotation-source population (`VAL-A`) was not generated, and no endpoint uses human labels.
- **Stub cost is zero:** stubs make no LLM calls, so `mean_llm_calls` is 0 there. The budget in protocol section 8 uses the report's call formula on stub runs, scaled by 1.4. Live numbers must come from the pilot.
- **The pilot set has no no-misconception controls.** Controls are every 10th candidate per job, so 12 candidates have none. False alarms on the pilot are measured on correct replies only.
- **The runner uses the code driver only.** The agent driver is demonstrated separately with `scripts/agent_demo.py` (Step 7, [step7_tools.md](step7_tools.md)).

## Findings from the stub checks (plumbing, not results)

On the 60 frozen candidates with stubs:

| Condition | Exchanges | Calls by formula | Skill accuracy | Recall |
| --- | ---: | ---: | ---: | ---: |
| B1 | 12.0 | 37.0 | 60.6% | — |
| ADAPT12 | 12.0 | 37.0 | 72.2% | — |
| A2 | 15.3 | 74.3 | 78.3% | 56% |
| A2Q | 15.9 | 76.2 | 67.8% | 40% |
| A2E | 15.9 | 93.9 | 67.8% | 40% |
| A4 | 15.3 | 54.1 | 78.3% | 56% |
| A5E | 15.5 | 92.3 | 70.6% | 32% |

- **Higher cost:** A2E costs about 18 more stub calls than A2Q at the same length, because it also diagnoses correct main answers.
- **Same accuracy drop as A2Q:** A2E keeps A2Q's lower stub accuracy. Questions chosen to corroborate a still-suspected misconception score 0 for its holder and pull the level estimate down (protocol section 2, stub warning). The stub diagnoser always supplies valid tag evidence, so no quote rejections or disputes appear here; those paths are covered by Souradeep's unit tests.

## Handoff to Souradeep

1. **Check that each control removes only the intended behaviour:**
   - ADAPT12: no diagnosis and no follow-ups (exactly 12 exchanges in `test_new_presets_run_end_to_end`);
   - A5E: only retrieval differs from A2E;
   - A4 is unchanged and stays separately named;
   - A2, A2Q and A2E stay distinct.
2. **Review the manifest's `PROMPT_SOURCES` list** in `src/part4/manifest.py`. Every engine source file is already in the fingerprint; the list only displays the prompt-relevant ones. Add any evidence file I missed.
3. **Step 6 (stopping):**
   - please look at the corroboration/scorecard interaction above;
   - add the per-skill posterior mass in the reported level band (protocol section 6);
   - give any stopping variant its own preset name, so it gets its own manifest entry.
4. **Say whether A2M is wanted.** If yes, we agree the tracker-reset design before either of us edits `followups.py` or `selection.py`.
