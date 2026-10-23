# Pending work

## Part 1: question bank and job data

The build is done: bank, misconceptions, job descriptions, validator, search index and tests. These items were deliberately skipped for later.

### Human review
- [ ] Review the 135 questions in `data/questions/` against spec section 13 and change `status` from `draft` to `checked`. The 45 added on 2026-10-23 are numbered 011–015 in every cell. In progress since 2026-10-23 on the shared review page (https://claude.ai/artifact/X1LazXWwYup2coDscLkebY, source and instructions in app/review/): 4 batches, verdict keep / needs a fix (with note) / drop. When done, read the verdicts, apply them, and refill any dropped questions from the 36 drafts left in `data/pool/` so every level stays the same size.
- [ ] Difficulty check (spec section 14): two people rate the same 21 questions blind.
- [ ] Verify the adapted citations:
    - [ ] the 4 data-structures misconceptions (Karpierz & Wolfman 2014; Gal-Ezer & Zur 2004), which were cited from summaries only
    - [ ] `sql` "= NULL treated like IS NULL": it mirrors the paper's finding rather than coming from it directly, so consider relabelling it "generated"

### Needs an API key
- [x] Switch `src/part1/extract_topics.py` from the Anthropic SDK to the OpenAI-compatible client, with a configurable `base_url` and model, so it works with Gemini, Groq or NVIDIA NIM.
- [x] Run `scripts/eval_topic_extraction.py`. Done 2026-10-19 on gemini-flash-lite-latest: recall 1.00 for all jobs, precision 0.96 (extra `sql-select-filter` for backend and data engineer). Use the same model as the interviewer. Watch the data-engineer job: its wording on transaction isolation may be read as `py-concurrency`.

## Part 2: interview engine

- [ ] Ask the TA what the course expects for "agentic" and "tool calling". This decides whether the LLM-agent driver is needed for the experiments (docs/part2_spec.md section 1).
- [x] Build the `agent` driver (done 2026-10-19). Still to do on the pilot: tune `uncertainty_threshold` and the level difficulties on the pilot.
- [ ] Pilot checks found in the stub dry runs:
    - very strong candidates hit the 15-question cap because the SE stays just above 0.6
    - a candidate right at level 1 / wrong at level 2 lands near θ = −0.3, which maps to level 2
- [x] Misconception recall: probing added (stub measurement: 23 → 32 of 80 planted confirmed). Still below the 60% target; revisit after real simulator runs. Growing the bank to 135 did not raise it (32 of 80 confirmed with probing; see part2_spec section "Selection and stopping"). Original note: a planted misconception is found only if a question listing it is asked (about 2 of 30 per skill list each one). See the options in the chat summary of 2026-10-16.
- [x] Streamlit app (live interviews): `app/streamlit_app.py`.
- [ ] Agent summary can contradict the decision (one live run called a needs-review candidate "strong potential"). Give the summary prompt the decision and levels.
- [ ] Quota: one live interview is about 70 LLM calls with the code driver, plus about 55 with the agent driver. Plan the experiments around the free-tier daily limit (cache, spread over days, agent driver on a subset only). Observed 2026-10-22: Gemini free tier about 20 s per call (2–3 min per answer); Groq `openai/gpt-oss-20b` about 1 s per call until its tokens-per-minute limit, then about 10 s (about 1,200 tokens per call). The app can switch provider (see docs/llm.md).
- [x] App: a second Submit while an answer is still grading started a second page run and crashed the interview (`NoneType has no attribute strip`). Fixed 2026-10-23 with `submit_once` in `app/streamlit_app.py`.
- [ ] Optional retrieval upgrade (decided later): rank the eligible questions by embedding similarity. (a) To the job description, for questions that fit the job better; (b) to the question and misconception the candidate just missed, to probe it again and raise recall. Today retrieval is a tag filter plus rule ranking and the embedding index is only used for duplicate checks. (b) can only be measured with the LLM simulator, not the stubs.

## Part 4: candidates and evaluation (built 2026-10-23, see docs/part4.md)

- [x] Evaluation definitions: written in docs/part4.md (Metrics) and coded in `src/part4/metrics.py`. Change them there before the live runs if the team disagrees.
- [ ] Choose the model and budget for the experiments: about 240 interviews (60 candidates × B1, A2, A4, A5) ≈ 20,000 LLM calls; the simulator must be a different model family from the grader.
- [x] Build: 60 candidate keys, the LLM simulator, the runner and the metrics; full stub run done (240 interviews).
- [ ] Two teammates check about 30 simulated answers against their plans, to confirm the simulator follows them.
- [ ] Live run: grader on Gemini (or Groq), candidates on Ollama llama3 (or Groq). Measured: llama3 is weak as a grader (2/6 misconceptions in the smoke test) but fine as a candidate.
- [ ] Pilot (about 20 interviews): tune `uncertainty_threshold`; in stub runs A2 asks about 12.9 main questions plus follow-ups, more than B1's 12.
- [ ] Optional small human study: (A) teammates act out secret candidate cards in the app (8 interviews, exact ground truth, real human answers); (B) people outside the team take it as themselves, compared with a human interviewer's rating. Needs a candidate-id field in the app and the Part 4 scoring code.

### Housekeeping
- [ ] Update the proposal's Data section: misconceptions are adapted from published catalogs, with citations, and no private hiring data is used.
- [ ] Commit the work of 2026-10-22/23 (the repo is under git with 2 commits; Groq provider, double-submit fix and the 135-question bank are uncommitted).
