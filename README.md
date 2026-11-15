# Adaptive AI Technical Interviewer

An AI interviewer for junior technical screening in **Python, SQL and data structures**.

- **Asks only what it needs.** It chooses each question to reduce its uncertainty about the candidate's level.
- **Stops when confident.** It ends the interview once it has enough evidence.
- **Explains wrong answers.** It works out why an answer was wrong, including which misconception the candidate holds.
- **Leaves the decision to a human.** It recommends **fit / not fit / needs human review** and never makes the hiring decision itself.

## Setup

**1. Install** (Python 3.12):

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # macOS / Linux
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
```

The commands below use the macOS / Linux path `.venv/bin/python`; on Windows use `.venv/Scripts/python.exe`.

**2. Add an API key.** Copy `.env.example` to `.env` and put your own Gemini key in it as `GEMINI_API_KEY` (and a Groq key as `GROQ_API_KEY` if you use Groq). Get one at https://aistudio.google.com/apikey. Never commit `.env`.

**3. Check everything works:**

```bash
.venv/bin/python -m pytest
```

## Run

| What | Command |
| --- | --- |
| Live interview in the browser | `.venv/bin/python -m streamlit run app/streamlit_app.py` |
| One simulated interview, offline (no key) | `.venv/bin/python scripts/run_interview.py --system A2E --job data-analyst` |
| The tool-calling agent, step by step, with rule-breaking calls refused (offline) | `.venv/bin/python scripts/agent_demo.py --system A2EV --probe-rules` |
| Validate the question bank | `.venv/bin/python scripts/validate_bank.py` |
| Rebuild the search index after editing questions | `.venv/bin/python scripts/build_index.py` |
| Add the best kept-back drafts from `data/pool/` to the bank (dry run without `--write`) | `.venv/bin/python scripts/promote_pool.py` |
| Generate the 60 simulated candidates | `.venv/bin/python scripts/make_candidates.py` |
| Run the experiment (stubs by default; see *Reproduce the final results* for the live run) | `.venv/bin/python scripts/run_experiment.py` |
| Metrics and targets for a run | `.venv/bin/python scripts/evaluate.py --run stub-stub` |
| Check the live grader and diagnoser | `.venv/bin/python scripts/smoke_llm.py` |
| Evaluate topic extraction | `.venv/bin/python scripts/eval_topic_extraction.py` |

## Systems compared

| System | What it is |
| --- | --- |
| `B1` | Fixed interview: 12 questions, no follow-ups, no diagnosis |
| `ADAPT12` | Adaptive question choice, exactly 12 questions, no diagnosis (ablation: adaptivity alone) |
| `A2E` | Full system: adaptive choice, early stopping, theory of mind of the candidate |
| `A4` | A2E without the theory of mind: one diagnosis, no quotes, one-reply confirmation (ablation) |
| `A5E` | A2E with job-blind question choice (ablation) |

Older presets (`A1`–`A5`, `A2EB`, `A2EV`) are kept in `src/shared/interfaces.py` for the mid-term runs and demos.

## Reproduce the final results

The final report (`report/final_report.pdf`) uses the run `reviewed-main-v2`: all five systems on the
same simulated candidates, one candidate at a time. The 22 candidates that finished in every system
are committed in `results/reviewed-main-v2-final22/` (one JSON per interview plus `manifest.json`,
which fingerprints the code, bank, prompts and models used).

**Regenerate every number and figure in the report** (no API key needed):

```bash
.venv/bin/python report/make_final_figures.py --run reviewed-main-v2-final22
cd report && tectonic final_report.tex
```

This writes `report/final_numbers.tex` and `report/figures/final_*.pdf`; the report never carries a
hand-copied number. For the full metric table:

```bash
.venv/bin/python scripts/evaluate.py --run reviewed-main-v2-final22 --method A2E --baseline B1 --comparisons
```

**Rerun the interviews** (needs Gemini and Groq keys in `.env`; grader `gemini-3.5-flash-lite`,
candidate `openai/gpt-oss-120b` on Groq):

```bash
.venv/bin/python scripts/run_rotating.py --grader-keys main --simulator-keys main \
    --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b \
    --systems B1 A2E A4 ADAPT12 A5E --by-candidate docs/final_run_order.txt --run reviewed-main-v2
```

`run_rotating.py` moves on to the next key when one hits its quota; list several key names to spread
the run. LLM responses are cached in `.cache/`, so an interrupted run resumes where it stopped.

## Layout

| Path | Contents |
| --- | --- |
| `data/` | Question bank (`questions/`), misconceptions, topics, jobs, simulated candidates (`candidates/`) |
| `data/pool/` | Both question batches and the blind selection record |
| `schemas/` | JSON formats: question, misconception, candidate key, interview result |
| `src/part1/` | Bank loading, validator, search index, topic extraction |
| `src/engine/` | Interview engine: scorecard (Partial Credit Model + EAP), question selection, follow-ups, report, tools, agent driver, session |
| `src/part3/` | Prompted grader and diagnoser |
| `src/part4/` | Simulated candidates (keys, simulator, respondent), experiment runner, metrics |
| `src/llm/` | LLM client for any OpenAI-compatible provider, with caching and retries |
| `src/shared/` | Interfaces between the parts, config and system presets (B1, ADAPT12, A2E, A4, A5E and older ones), stubs for offline testing |
| `app/` | Streamlit app |
| `examples/` | One example candidate key and one interview result |
| `results/` | Run outputs (gitignored), except the final run `reviewed-main-v2-final22/` |
| `report/` | Final and mid-term reports, and the scripts that generate their numbers and figures |
| `tests/` | Test suite (no network needed) |

## Docs

| Doc | Covers |
| --- | --- |
| `docs/interfaces.md` | Start here: how the parts connect |
| `docs/part1_spec.md` | Question bank decisions |
| `docs/part1_code.md` | How to use the bank code |
| `docs/part2_spec.md` | Engine design |
| `docs/part2_code.md` | How to run the engine |
| `docs/llm.md` | Models, keys, quotas |
| `docs/part4.md` | Candidates, experiment runner and metrics |
| `docs/todo.md` | What is pending |

## Status

| Part | Status |
| --- | --- |
| 1. Question bank and job data | Done: 135 questions, all reviewed by the team and checked. |
| 2. Interview engine | Done: scorecard, adaptive selection, stopping, follow-ups, tool-calling agent. |
| 3. Grader and diagnosis | Done: prompted grader and theory-of-mind diagnoser (votes, quote check, belief ledger). |
| 4. Candidates and evaluation | Done: final live run on 22 paired candidates x 5 systems (110 interviews); see `report/final_report.pdf`. |

Not done: human validation of the grades, and the remaining 38 of the 60 candidates (see `docs/todo.md`).

**Cost note:** grading and diagnosis use `gemini-3.5-flash-lite`. A full-system (A2E) interview makes
about 113 interviewer-side LLM calls, the fixed interview about 37. Responses are cached in `.cache/`,
so re-runs are free.
