r"""Interview app: a person takes the adaptive interview in the browser.

    .venv\Scripts\python.exe -m streamlit run app\streamlit_app.py

Sidebar: job, system preset, live (Gemini, Groq or local Ollama) or offline stub mode, interviewer view, start.
The logic lives in plain functions at the top of this file (tested in tests/test_app.py);
`main()` only draws the page.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from engine.factory import FactoryError, build_components, live_status  # noqa: E402
from engine.result import save_result  # noqa: E402
from engine.session import InterviewSession  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from shared.interfaces import SYSTEMS  # noqa: E402

LIVE_MODE = "Live grading (Gemini)"
GROQ_MODE = "Live grading (Groq)"
OLLAMA_MODE = "Live grading (Ollama, local)"
STUB_MODE = "Offline stub"
MODES = (LIVE_MODE, GROQ_MODE, OLLAMA_MODE, STUB_MODE)
MODE_PROVIDERS = {LIVE_MODE: "gemini", GROQ_MODE: "groq", OLLAMA_MODE: "ollama"}   # live modes -> llm.providers name
DEFAULT_SYSTEM = "A2"
LIVE_DIR = Path(os.environ.get("INTERVIEW_RESULTS_DIR") or ROOT / "results" / "live")

SKILL_NAMES = {"py": "Python", "sql": "SQL", "ds": "Data structures"}
DECISION_LABELS = {"fit": "Fit", "not_fit": "Not fit", "needs_review": "Needs review"}
MEETS_LABELS = {True: "Meets", False: "Below", None: "Uncertain"}
STUB_OUTCOMES = {"(type the tag myself)": None, "correct": "correct", "partial": "partial", "wrong": "wrong"}


# --------------------------------------------------------------------------
# Plain functions (no Streamlit)
# --------------------------------------------------------------------------

def load_jobs(root: Path = ROOT) -> list[dict]:
    """The jobs of data/jobs.json."""
    return json.loads((Path(root) / "data" / "jobs.json").read_text(encoding="utf-8"))


def table_frame(table: dict) -> pd.DataFrame:
    """A question's table as a small dataframe. Cells are text (NULL for None), so mixed
    columns display without conversion errors."""
    rows = [["NULL" if cell is None else str(cell) for cell in row] for row in table["rows"]]
    return pd.DataFrame(rows, columns=[str(c) for c in table["columns"]])


def question_parts(question: dict) -> dict:
    """What to draw for a question: its text word for word, the code and its language, the tables."""
    code = question.get("code")
    return {"text": question["question"], "code": code,
            "language": "sql" if question.get("skill") == "sql" else "python",
            "tables": [(t["name"], table_frame(t)) for t in question.get("tables") or []]}


def with_stub_tag(answer: str, outcome: str | None) -> str:
    """Offline stub mode: the stub grader reads a hidden tag from the answer. This adds one."""
    return answer if not outcome else f"{answer}\n[[sim outcome={outcome}]]".strip()


def make_config(system: str, seed: int = 0):
    return SYSTEMS[system].with_(seed=int(seed))


def start_session(job: dict, system: str, live: bool, seed: int = 0, root: Path = ROOT,
                  provider: str = "gemini") -> InterviewSession:
    """A new interview session. Live mode with no API key raises FactoryError with a clear message.
    `provider` (live only) is a name from `llm.providers.PROVIDERS`."""
    config = make_config(system, seed)
    if live:
        ready, message = live_status(provider)
        if not ready:
            raise FactoryError(message)
        from llm.providers import apply_provider
        config = apply_provider(config, provider)
    components = build_components(config, live=live, root=root, provider=provider if live else None)
    return InterviewSession(job, config, components)


def turn_rows(session: InterviewSession, interviewer: bool) -> list[dict]:
    """Previous turns for the history: question, answer, follow-ups; grade and diagnosis for the interviewer."""
    rows = []
    for turn in session.turns:
        row = {"turn": turn["index"], "skill": SKILL_NAMES[turn["skill"]], "level": turn["level"],
               "question": turn["prompt"].split("\n\n")[0], "answer": turn["answer"],
               "follow_ups": [{"prompt": f["prompt"], "answer": f["answer"]} for f in turn["follow_ups"]]}
        if interviewer:
            diagnosis = turn["diagnosis"]
            row["grade"] = turn["grade"]["score"]
            row["votes"] = turn["grade"]["votes"]
            row["diagnosis"] = None if not diagnosis else (
                diagnosis["cause"] + (f" ({diagnosis['misconception_id']})" if diagnosis.get("misconception_id") else ""))
        rows.append(row)
    return rows


def scorecard_frame(view: dict) -> pd.DataFrame:
    """The live scorecard (session.scorecard_view()) as a table."""
    return pd.DataFrame([{"Skill": SKILL_NAMES[s], "Questions": r["n"], "Estimate": round(r["estimate"], 2),
                          "SE": round(r["se"], 2), "Level": r["level"], "Job minimum": r["minimum"],
                          "SE below threshold": "yes" if r["confident"] else "no"}
                         for s, r in view.items()])


def misconception_rows(findings: list[dict], records: dict) -> list[dict]:
    rows = []
    for finding in findings:
        record = records.get(finding["id"], {})
        rows.append({"id": finding["id"], "description": record.get("description", finding["id"]),
                     "status": "Confirmed" if finding["confirmed"] else "Suspected",
                     "turns": finding.get("evidence_turns", [])})
    return rows


def result_overview(result: dict, records: dict) -> dict:
    """The final screen as plain data: decision, per-skill table, misconceptions, coverage, summary."""
    report = result["report"]
    skills = [{"Skill": SKILL_NAMES[s], "Level": r["level"], "Estimate": r.get("estimate"), "SE": r.get("se"),
               "Job minimum": r["minimum"], "Meets minimum": MEETS_LABELS[r.get("meets_minimum")]}
              for s, r in report["levels"].items()]
    coverage = report["topic_coverage"]
    return {"decision": DECISION_LABELS[report["decision"]], "reason": report.get("decision_reason", ""),
            "skills": skills, "misconceptions": misconception_rows(report["misconceptions"], records),
            "tested": coverage["tested"], "untested": coverage["untested"], "required": coverage["required"],
            "summary": report.get("summary", ""), "flags": report.get("flags", []),
            "errors": result.get("errors", []), "stop_reason": result["stop_reason"],
            "questions": len(result["turns"]), "exchanges": result["exchanges"]}


def result_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)


def save_live_result(result: dict, folder: Path = LIVE_DIR) -> Path:
    """results/live/live-<timestamp>.json"""
    return save_result(result, folder=folder)


def submit_once(state, session: InterviewSession, prompt: dict, text: str) -> bool:
    """Submit `text` as the answer to `prompt`; False (nothing sent) if another answer is still
    being graded or the interview has already moved past `prompt`.

    Pressing Submit again while grading makes Streamlit start a second run of the page next to
    the first, and the session must never be used by two runs at once."""
    lock = state.setdefault("submit_lock", threading.Lock())
    if not lock.acquire(blocking=False):
        return False
    try:
        if session.next_prompt() != prompt:
            return False
        session.submit(text)
        return True
    finally:
        lock.release()


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def draw_question(st, parts: dict) -> None:
    st.markdown(parts["text"])
    if parts["code"]:
        st.code(parts["code"], language=parts["language"])
    for name, frame in parts["tables"]:
        st.caption(f"Table {name}")
        st.dataframe(frame, hide_index=True)


def draw_interviewer_panel(st, session: InterviewSession) -> None:
    view = session.interviewer_view()
    st.subheader("Interviewer view")
    st.dataframe(scorecard_frame(view["scorecard"]), hide_index=True)
    current = view["current"]
    if current:
        grade, diagnosis = current["provisional_grade"], current["diagnosis"]
        st.write(f"Provisional grade of this answer: **{grade['score']}** (votes {grade['votes']})")
        if diagnosis:
            line = f"Diagnosis: **{diagnosis['cause']}**"
            if diagnosis.get("misconception_id"):
                line += f" ({diagnosis['misconception_id']})"
            st.write(line)
    rows = misconception_rows(view["misconceptions"], session.components.misconceptions)
    if rows:
        st.write("Misconceptions so far")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    for flag in view["flags"]:
        st.warning(flag)


def draw_history(st, session: InterviewSession, interviewer: bool) -> None:
    rows = turn_rows(session, interviewer)
    if not rows:
        return
    st.subheader("Previous questions")
    for row in reversed(rows):
        label = f"Question {row['turn']} ({row['skill']}, level {row['level']})" if interviewer \
            else f"Question {row['turn']}"
        with st.expander(label):
            st.markdown(row["question"])
            st.write("**Answer:**", row["answer"])
            for follow_up in row["follow_ups"]:
                st.write("**Follow-up:**", follow_up["prompt"])
                st.write("**Answer:**", follow_up["answer"])
            if interviewer:
                st.caption(f"Final grade {row['grade']} (votes {row['votes']}); diagnosis: {row['diagnosis'] or '-'}")


def draw_final(st, session: InterviewSession) -> None:
    state = st.session_state
    result = state["result"]
    overview = result_overview(result, session.components.misconceptions)
    st.header("Interview finished")
    decision = overview["decision"]
    (st.success if decision == "Fit" else st.error if decision == "Not fit" else st.warning)(
        f"Decision: {decision}. {overview['reason']}")
    st.subheader("Skill levels")
    st.dataframe(pd.DataFrame(overview["skills"]), hide_index=True)
    st.subheader("Misconceptions")
    if overview["misconceptions"]:
        st.dataframe(pd.DataFrame(overview["misconceptions"]), hide_index=True)
    else:
        st.write("None found.")
    st.subheader("Topic coverage")
    st.write(f"Tested {len(overview['tested'])} of {len(overview['required'])} required topics.")
    if overview["untested"]:
        st.write("Not tested: " + ", ".join(overview["untested"]))
    st.subheader("Summary")
    st.write(overview["summary"])
    for flag in overview["flags"]:
        st.warning(flag)
    for error in overview["errors"]:
        st.error(error)
    st.caption(f"Stop reason: {overview['stop_reason']}; {overview['questions']} questions, "
               f"{overview['exchanges']} exchanges.")
    st.download_button("Download result (JSON)", result_json(result), file_name=Path(state["saved_path"]).name,
                       mime="application/json")
    st.caption(f"Saved to {state['saved_path']}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Adaptive technical interview", layout="wide")
    state = st.session_state
    bank = load_bank()
    jobs = load_jobs()

    with st.sidebar:
        st.title("Interview setup")
        job = st.selectbox("Job", jobs, format_func=lambda j: j["title"])
        with st.expander("Job description"):
            st.write(job["description"])
            st.caption("Minimum levels: " + ", ".join(f"{SKILL_NAMES[s]} {m}" for s, m in job["min_levels"].items()))
        system = st.selectbox("System preset", list(SYSTEMS), index=list(SYSTEMS).index(DEFAULT_SYSTEM))
        mode = st.radio("Mode", MODES, index=MODES.index(STUB_MODE))
        provider = MODE_PROVIDERS.get(mode)
        live = provider is not None
        interviewer = st.toggle("Show interviewer view", value=False)
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
        if live:
            ready, message = live_status(provider)
            (st.caption if ready else st.warning)(message)
        start = st.button("Start / Restart", type="primary")

    if start:
        for key in ("session", "result", "saved_path", "error"):
            state.pop(key, None)
        try:
            state["session"] = start_session(job, system, live, seed, provider=provider or "gemini")
        except FactoryError as e:
            state["error"] = str(e)
        except Exception as e:
            state["error"] = f"Could not start the interview ({type(e).__name__}: {e})"

    st.title("Adaptive technical interview")
    if state.get("error"):
        st.error(state["error"])
    session: InterviewSession | None = state.get("session")
    if session is None:
        st.write("Choose a job and a mode in the sidebar, then press **Start / Restart**.")
        return

    if session.finished:
        if "result" not in state:
            state["result"] = session.result()
            state["saved_path"] = str(save_live_result(state["result"]))
        draw_final(st, session)
        if interviewer:
            draw_history(st, session, True)
        return

    left, right = st.columns([3, 2]) if interviewer else (st.container(), None)
    with left:
        prompt = session.next_prompt()
        if prompt["kind"] == "follow_up":
            st.info("Follow-up question")
            exchange = session.current_exchange
            if exchange:
                st.caption("Your answer so far: " + exchange["answer"])
            st.markdown(prompt["text"])
            with st.expander("The question this follows"):
                draw_question(st, question_parts(prompt["question"]))
        else:
            st.subheader(f"Question {prompt['turn']}")
            if interviewer:
                st.caption(f"{SKILL_NAMES[prompt['skill']]}, level {prompt['level']}")
            draw_question(st, question_parts(prompt["question"]))

        stub_choice = None
        if not live:
            st.caption("Offline stub mode: answers are graded by a hidden tag, not by reading them. "
                       "Pick how this answer should count, or type a tag such as [[sim outcome=partial]].")
            stub_choice = st.selectbox("Stub grade for this answer", list(STUB_OUTCOMES), key="stub_outcome")
        with st.form("answer_form", clear_on_submit=True):
            answer = st.text_area("Your answer", height=160)
            submitted = st.form_submit_button("Submit")
        if submitted:
            text = answer if live else with_stub_tag(answer, STUB_OUTCOMES[stub_choice])
            with st.spinner("Grading your answer…"):
                sent = submit_once(state, session, prompt, text)
            if sent:
                st.rerun()
            st.warning("That answer was not sent: the previous one is still being graded. Please wait.")
        draw_history(st, session, interviewer)
    if right is not None:
        with right:
            draw_interviewer_panel(st, session)


if __name__ == "__main__":
    main()
