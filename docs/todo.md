# Pending work

## Part 1: question bank and job data

The build is done: bank, misconceptions, job descriptions, validator, search index and tests. These items were deliberately skipped for later.

### Human review
- [ ] Review the 90 questions in `data/questions/` against spec section 13 and change `status` from `draft` to `checked`.
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
- [x] Misconception recall: probing added (stub measurement: 23 → 32 of 80 planted confirmed). Still below the 60% target; revisit after real simulator runs. Original note: a planted misconception is found only if a question listing it is asked (about 2 of 30 per skill list each one). See the options in the chat summary of 2026-10-16.
- [x] Streamlit app (live interviews): `app/streamlit_app.py`.
- [ ] Agent summary can contradict the decision (one live run called a needs-review candidate "strong potential"). Give the summary prompt the decision and levels.
- [ ] Quota: one live interview is about 70 LLM calls with the code driver, plus about 55 with the agent driver. Plan the experiments around the free-tier daily limit (cache, spread over days, agent driver on a subset only).

### Housekeeping
- [ ] Update the proposal's Data section: misconceptions are adapted from published catalogs, with citations, and no private hiring data is used.
- [ ] Put the project under git.
