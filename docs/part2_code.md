# Part 2 code: interview engine, app and agent driver

All commands run from the project root with the project virtual env (Python 3.12). The rules are in `docs/part2_spec.md`.

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

## Layout

| Path | What it does |
| --- | --- |
| `src/engine/scorecard.py`, `selection.py`, `followups.py`, `report.py`, `tools.py` | Scoring, question choice, turn flow, report, rule-enforcing tools |
| `src/engine/session.py` | `InterviewSession`: the step-based interview |
| `src/engine/run.py` | `run_interview()`: blocking loop over a session (code driver); routes `driver="agent"` |
| `src/engine/agent.py` | LLM-agent driver |
| `src/engine/factory.py` | `build_components(config, live=...)` |
| `src/engine/result.py` | Result assembly and `save_result` |
| `app/streamlit_app.py` | Web app |
| `scripts/run_interview.py`, `scripts/misconception_recall.py` | Command-line runs |

## CLI: one simulated interview

```
.venv\Scripts\python.exe scripts\run_interview.py --system A2 --seed 0
.venv\Scripts\python.exe scripts\run_interview.py --system B1 --candidate examples\candidate.example.json --save
```

A rule-based candidate (from a candidate key) answers; the stub grader and diagnoser read its hidden tags. No API key. Systems: B1, A1 to A5. `--save` writes `results/<system>/<job>/<candidate>.json`.

Misconception recall with probing off and on (10 generated candidates, `--seeds N` repeats each):

```
.venv\Scripts\python.exe scripts\misconception_recall.py
```

## From code

```python
from engine.factory import build_components
from engine.session import InterviewSession
from part1.bank import load_bank
from shared.interfaces import SYSTEMS

config = SYSTEMS["A2"].with_(seed=1)
components = build_components(config, live=False)          # live=True: Gemini grader and diagnoser
job = load_bank().get_job("data-analyst")
session = InterviewSession(job, config, components)
while (prompt := session.next_prompt()) is not None:
    print(prompt["kind"], prompt["text"])
    session.submit(input("> "))
result = session.result()
```

`run_interview(job, respondent, config, components)` does the same loop with a `Respondent` (a simulated candidate).

## The web app

```
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Sidebar:

- **Job:** from `data/jobs.json`; the description is in an expander.
- **System preset:** default A2.
- **Mode:** "Offline stub" (default, no key) or "Live grading (Gemini)". Live mode needs `LLM_BASE_URL` and `LLM_API_KEY` in the environment or in `.env`; without them the app shows a message and does not start.
- **Show interviewer view:** the live scorecard (estimate, SE, level), provisional grade and diagnosis of the current answer, misconceptions so far, and grades in the history.
- **Start / Restart.**

The main area shows the current question (text, code block, small tables), a "Follow-up question" label for follow-ups, the answer box and the history. At the end it shows the decision and reason, the per-skill levels and whether each meets the minimum, confirmed and suspected misconceptions, topic coverage, the summary and a download button. The result is saved to `results/live/live-<timestamp>.json`.

Offline stub mode grades by a hidden tag in the answer, not by reading it. The page has a "Stub grade for this answer" selector (correct, partial, wrong) that adds the tag, or type `[[sim outcome=partial]]` yourself.

## The agent driver

```python
config = SYSTEMS["A2"].with_(driver="agent")               # models.interviewer picks the model
components = build_components(config, live=True)           # needs an LLM client (components.llm)
result = run_interview(job, respondent, config, components)
```

An LLM calls `search_question_bank`, `ask_candidate`, `grade_answer`, `ask_follow_up`, `update_scorecard` and `end_interview`; the tools enforce the rules. `max_agent_steps` (default 150) bounds the LLM calls. The result has an `agent` section (steps, refused calls, invalid calls, model) and every call in the turns' `tool_calls`.

The tests use a scripted fake LLM (`tests/test_engine_agent.py`); no live call has been made. To try it for real, pass a live `components` and a `Respondent` for the person or simulated candidate. The CLI script does not run the agent driver.

## Tests

| File | Covers |
| --- | --- |
| `tests/test_engine_selection.py`, `test_engine_interviews.py`, `test_engine_tools.py`, `test_engine_scorecard.py` | Engine rules, presets, probing, determinism |
| `tests/test_engine_session.py` | Session API; equals the old blocking loop for the same seed |
| `tests/test_engine_factory.py` | Stub components; live components with fake classes |
| `tests/test_app.py` | App helpers; headless run with `streamlit.testing.v1.AppTest` |
| `tests/test_engine_agent.py` | Agent driver with a scripted fake LLM |
