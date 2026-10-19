"""The Streamlit app: its plain helper functions, and a headless run with streamlit's AppTest
(no browser, no API key)."""

import importlib.util
import json

import pytest

from engine.factory import FactoryError
from engine_support import ROOT, STRONG, assert_valid, real_bank, run

APP = ROOT / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("streamlit_app_module", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)       # defines helpers only: the page is drawn by main()
    return module


# ---- helpers ---------------------------------------------------------------------------------

def test_jobs_are_loaded_from_data(app):
    jobs = app.load_jobs()
    assert [j["id"] for j in jobs] == [j["id"] for j in real_bank().jobs]
    assert all(j["description"] and j["min_levels"] for j in jobs)


def test_question_parts_show_text_code_and_small_tables(app):
    question = real_bank().get_question("SQL-L2-007")
    parts = app.question_parts(question)
    assert parts["text"] == question["question"] and parts["code"] == question["code"]
    assert parts["language"] == "sql"
    names = [name for name, _ in parts["tables"]]
    assert names == [t["name"] for t in question["tables"]]
    frame = parts["tables"][0][1]
    assert list(frame.columns) == question["tables"][0]["columns"] and len(frame) == len(question["tables"][0]["rows"])
    assert "NULL" in parts["tables"][1][1].to_string()       # None cells are shown as NULL
    plain = app.question_parts({"question": "Explain a hash table.", "skill": "ds"})
    assert plain["code"] is None and plain["tables"] == [] and plain["language"] == "python"


def test_stub_tag_is_added_only_when_asked(app):
    assert app.with_stub_tag("my answer", None) == "my answer"
    assert app.with_stub_tag("my answer", "partial") == "my answer\n[[sim outcome=partial]]"
    assert app.with_stub_tag("", "correct") == "[[sim outcome=correct]]"


def test_offline_session_runs_to_a_finished_result(app):
    job = app.load_jobs()[0]
    session = app.start_session(job, "A2", live=False, seed=1)
    answers = 0
    while (prompt := session.next_prompt()) is not None:
        session.submit(app.with_stub_tag(f"answer {answers}", "correct"))
        answers += 1
    result = session.result()
    assert result["system"] == "A2" and result["config"]["seed"] == 1 and result["candidate_id"] is None
    assert answers == len(result["turns"]) and result["report"]["decision"] in ("fit", "not_fit", "needs_review")
    assert_valid(result)
    overview = app.result_overview(result, session.components.misconceptions)
    assert overview["decision"] in ("Fit", "Not fit", "Needs review")
    assert [r["Skill"] for r in overview["skills"]] == ["Python", "SQL", "Data structures"]
    assert all(r["Meets minimum"] in ("Meets", "Below", "Uncertain") for r in overview["skills"])
    assert json.loads(app.result_json(result)) == result


def test_live_mode_without_a_key_gives_a_clear_message(app, monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("llm.client.load_env", lambda path=None: {})
    with pytest.raises(FactoryError, match="API key"):
        app.start_session(app.load_jobs()[0], "A2", live=True)


def test_turn_rows_hide_grades_unless_the_interviewer_view_is_on(app):
    job = app.load_jobs()[0]
    session = app.start_session(job, "A2", live=False)
    session.submit("first [[sim outcome=wrong]]")
    session.submit("[[sim outcome=wrong]]") if session.next_prompt() else None
    candidate_rows = app.turn_rows(session, interviewer=False)
    interviewer_rows = app.turn_rows(session, interviewer=True)
    assert candidate_rows and "grade" not in candidate_rows[0] and "diagnosis" not in candidate_rows[0]
    assert interviewer_rows[0]["grade"] == 0 and interviewer_rows[0]["diagnosis"].startswith("gap")
    frame = app.scorecard_frame(session.scorecard_view())
    assert list(frame["Skill"]) == ["Python", "SQL", "Data structures"] and frame["Questions"].sum() == len(candidate_rows)


def test_misconceptions_are_listed_with_descriptions_and_status(app):
    records = real_bank().misconception_by_id
    mid = "sql.joins.left-join-drops-unmatched"
    rows = app.misconception_rows([{"id": mid, "confirmed": True, "evidence_turns": [2, 4]},
                                   {"id": "py.collections.tuple-mutable", "confirmed": False, "evidence_turns": [3]}],
                                  records)
    assert rows[0]["status"] == "Confirmed" and rows[0]["description"] == records[mid]["description"]
    assert rows[1]["status"] == "Suspected" and rows[1]["turns"] == [3]


def test_live_results_are_saved_under_results_live(app, tmp_path):
    result, _ = run(STRONG, "B1")
    result = dict(result, candidate_id=None, started_at="2026-10-20T10:04:12Z")
    path = app.save_live_result(result, tmp_path / "live")
    assert path == tmp_path / "live" / "live-20261020100412.json"
    assert json.loads(path.read_text(encoding="utf-8")) == result


# ---- the whole page, headless ----------------------------------------------------------------

def click(at, label):
    next(b for b in at.button if b.label == label).click()


def test_app_runs_offline_without_an_api_key(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("INTERVIEW_RESULTS_DIR", str(tmp_path))
    for name in ("LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert [s.label for s in at.sidebar.selectbox][:2] == ["Job", "System preset"]
    assert at.sidebar.selectbox[1].value == "A2"                      # default system
    assert at.sidebar.radio[0].value == "Offline stub"                # works without a key

    click(at, "Start / Restart")
    at.run()
    assert not at.exception and not at.error
    assert any("Question 1" in s.value for s in at.subheader)
    assert at.text_area[0].label == "Your answer"

    for i in range(40):                                                # answer until it finishes
        if any(h.value == "Interview finished" for h in at.header):
            break
        at.selectbox(key="stub_outcome").select("correct")
        at.text_area[0].input(f"answer {i}")
        click(at, "Submit")
        at.run()
        assert not at.exception
    assert any(h.value == "Interview finished" for h in at.header)
    assert any(s.value.startswith("Decision:") for s in [*at.success, *at.error, *at.warning])
    saved = list(tmp_path.glob("live-*.json"))
    assert len(saved) == 1
    assert_valid(json.loads(saved[0].read_text(encoding="utf-8")))


def test_app_shows_a_clear_message_for_live_mode_without_a_key(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("INTERVIEW_RESULTS_DIR", str(tmp_path))
    for name in ("LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("llm.client.load_env", lambda path=None: {})
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Live grading (Gemini)")
    click(at, "Start / Restart")
    at.run()
    assert not at.exception
    assert any("API key" in e.value for e in at.error)
    assert not at.text_area                                            # no interview was started


def test_interviewer_view_toggle_shows_the_scorecard(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("INTERVIEW_RESULTS_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.toggle[0].set_value(True)
    click(at, "Start / Restart")
    at.run()
    assert not at.exception
    assert any(s.value == "Interviewer view" for s in at.subheader)
    assert at.dataframe                                                # the live scorecard
