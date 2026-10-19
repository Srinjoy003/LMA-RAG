# Adaptive AI Technical Interviewer

An AI interviewer for junior technical screening in **Python, SQL and data structures**.

- **Asks only what it needs.** It chooses each question to reduce its uncertainty about the candidate's level.
- **Stops when confident.** It ends the interview once it has enough evidence.
- **Explains wrong answers.** It works out why an answer was wrong, including which misconception the candidate holds.
- **Leaves the decision to a human.** It recommends **fit / not fit / needs human review** and never makes the hiring decision itself.

## Setup

**1. Install** (Windows, Python 3.12):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**2. Add an API key.** Copy `.env.example` to `.env` and put your own free Gemini key in it. Get one at https://aistudio.google.com/apikey. Never commit `.env`.

**3. Check everything works:**

```bash
.venv/Scripts/python.exe -m pytest
```

## Run

| What | Command |
| --- | --- |
| Live interview in the browser | `.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py` |
| One simulated interview, offline (no key) | `.venv/Scripts/python.exe scripts/run_interview.py --system A2 --job data-analyst` |
| Validate the question bank | `.venv/Scripts/python.exe scripts/validate_bank.py` |
| Rebuild the search index after editing questions | `.venv/Scripts/python.exe scripts/build_index.py` |
| Check the live grader and diagnoser | `.venv/Scripts/python.exe scripts/smoke_llm.py` |
| Evaluate topic extraction | `.venv/Scripts/python.exe scripts/eval_topic_extraction.py` |

## Layout

| Path | Contents |
| --- | --- |
| `data/` | Question bank (`questions/`), misconceptions, topics, jobs |
| `data/pool/` | Both question batches and the blind selection record |
| `schemas/` | JSON formats: question, misconception, candidate key, interview result |
| `src/part1/` | Bank loading, validator, search index, topic extraction |
| `src/engine/` | Interview engine: scorecard (Partial Credit Model + EAP), question selection, follow-ups, report, tools, agent driver, session |
| `src/part3/` | Prompted grader and diagnoser |
| `src/llm/` | LLM client for any OpenAI-compatible provider, with caching and retries |
| `src/shared/` | Interfaces between the parts, config and system presets (B1, A1–A5), stubs for offline testing |
| `app/` | Streamlit app |
| `examples/` | One example candidate key and one interview result |
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
| `docs/todo.md` | What is pending |

## Status

| Part | Status |
| --- | --- |
| 1. Question bank and job data | Built. Human review of the 90 questions is pending. |
| 2. Interview engine | Built and tested live. Pilot tuning is pending. |
| 3. Grader and diagnosis | Prompted grader and diagnoser work. The fine-tuned grader is not started. |
| 4. Candidates and evaluation | Not started: 60 candidate keys, simulator, experiment runner, metrics. |

**Quota note:** every role uses `gemini-3.5-flash-lite` (free tier). One live interview is about 70 LLM calls. Responses are cached in `.cache/`, so re-runs are free.
