"""The step-based session API (spec section 1) and its equivalence with the old blocking loop."""

import json
import random

import pytest

from engine import report
from engine.followups import run_exchange
from engine.result import build_result, build_turn, now, required_topics_for
from engine.run import run_interview
from engine.selection import InterviewState, check_stop
from engine.session import InterviewSession
from engine.tools import InterviewTools
from shared.interfaces import SYSTEMS
from shared.stubs import RuleBasedRespondent

from engine_support import (EXAMPLE_CANDIDATE, STRONG, WEAK, assert_valid, job_for, make_candidate,
                            make_components, misconception_setup, run, without_timestamps)


def legacy_run_interview(job, respondent, config, components, *, candidate_id=None, rng=None):
    """The blocking loop as it was before the session existed (kept here as the reference)."""
    rng = rng or random.Random(config.seed)
    started = now()
    errors = []
    required = required_topics_for(job, config, components, errors)
    state = InterviewState(job, config, components, required, rng)
    state.errors = errors
    tools = InterviewTools(state, components.grader)
    known_ids = set(components.misconceptions)
    turns = []
    try:
        while check_stop(state) is None:
            found = tools.search_question_bank()
            if not found["ok"]:
                raise RuntimeError(f"search_question_bank refused: {found['error']}")
            question, n = state.pending, found["turn"]
            answer = respondent.reply(found["prompt"], question=question, kind="main", turn=n)
            exchange = run_exchange(question, found["prompt"], answer, turn=n, respondent=respondent,
                                    diagnoser=components.diagnoser, config=config, tracker=state.tracker,
                                    grade=tools.grade, known_ids=known_ids)
            update = tools.update_scorecard(question["id"], answer, exchange["follow_ups"])
            if not update["ok"]:
                raise RuntimeError(f"update_scorecard refused: {update['error']}")
            turns.append(build_turn(found, answer, exchange, update))
        ended = tools.end_interview()
        if not ended["ok"]:
            raise RuntimeError(f"end_interview refused: {ended['error']}")
        stop_reason, final_report = tools.stop_reason, tools.report
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
        stop_reason, final_report = "error", report.build_report(state)
    return build_result(job=job, config=config, candidate_id=candidate_id, started=started, finished=now(),
                        required=required, turns=turns, stop_reason=stop_reason, report=final_report,
                        errors=errors, tool_log=tools.log)


def drive(session, respondent):
    """A caller that answers the session's prompts, one by one."""
    prompts = []
    while (prompt := session.next_prompt()) is not None:
        prompts.append(prompt)
        session.submit(respondent.reply(prompt["text"], question=prompt["question"],
                                        kind=prompt["kind"], turn=prompt["turn"]))
    return prompts


def new_session(candidate, system="A2", seed=0, job=None, components=None, **changes):
    config = SYSTEMS[system].with_(seed=seed, **changes)
    return InterviewSession(job or job_for(candidate), config, components or make_components(),
                            candidate_id=candidate["id"], rng=random.Random(seed))


# ---- same results as before --------------------------------------------------------------

@pytest.mark.parametrize("system", ["B1", "A1", "A2", "A4", "A5"])
@pytest.mark.parametrize("candidate", [EXAMPLE_CANDIDATE, STRONG, WEAK,
                                       make_candidate(3, misread_rate=1.0, id="CAND-MISREAD")],
                         ids=["example", "strong", "weak", "misread"])
def test_session_gives_the_same_result_as_the_old_blocking_loop(system, candidate):
    for seed in (0, 5):
        config = SYSTEMS[system].with_(seed=seed)
        old = legacy_run_interview(job_for(candidate), RuleBasedRespondent(candidate), config,
                                   make_components(), candidate_id=candidate["id"], rng=random.Random(seed))
        session = InterviewSession(job_for(candidate), config, make_components(),
                                   candidate_id=candidate["id"], rng=random.Random(seed))
        drive(session, RuleBasedRespondent(candidate))
        assert without_timestamps(session.result()) == without_timestamps(old)
        new, _ = run(candidate, system, seed=seed)
        assert without_timestamps(new) == without_timestamps(old)


def test_session_matches_the_old_loop_for_a_planted_misconception():
    candidate, job, source, _ = misconception_setup("py.variables-mutability.assignment-copies")
    config = SYSTEMS["A2"].with_(seed=2)
    old = legacy_run_interview(job, RuleBasedRespondent(candidate), config, make_components(source),
                               candidate_id=candidate["id"], rng=random.Random(2))
    session = InterviewSession(job, config, make_components(source), candidate_id=candidate["id"],
                               rng=random.Random(2))
    drive(session, RuleBasedRespondent(candidate))
    assert without_timestamps(session.result()) == without_timestamps(old)
    assert old["report"]["misconceptions"]


# ---- the prompts -----------------------------------------------------------------------------

def test_prompts_have_kind_turn_text_question_skill_and_level():
    candidate = make_candidate(3, misread_rate=1.0, id="CAND-MISREAD")      # every main answer gets a follow-up
    session = new_session(candidate, max_questions=3)
    prompts = drive(session, RuleBasedRespondent(candidate))
    result = session.result()
    assert [p["kind"] for p in prompts] == ["main", "follow_up"] * 3
    for prompt in prompts:
        assert set(prompt) == {"kind", "turn", "text", "question", "skill", "level"}
        turn = result["turns"][prompt["turn"] - 1]
        assert (prompt["skill"], prompt["level"]) == (turn["skill"], turn["level"])
        assert prompt["question"]["id"] == turn["question_id"]
        if prompt["kind"] == "main":
            assert prompt["text"] == turn["prompt"]
        else:
            assert prompt["text"] == turn["follow_ups"][0]["prompt"]
    assert [p["turn"] for p in prompts] == [1, 1, 2, 2, 3, 3]


def test_next_prompt_asks_the_same_question_until_it_is_answered():
    session = new_session(STRONG)
    first = session.next_prompt()
    assert session.next_prompt() == first and not session.finished
    session.submit("[[sim outcome=correct]]")
    assert session.next_prompt()["turn"] == 2 and session.next_prompt()["question"]["id"] != first["question"]["id"]


def test_prompt_question_is_a_copy_for_display():
    session = new_session(STRONG)
    prompt = session.next_prompt()
    prompt["question"]["question"] = "tampered"
    session.submit("[[sim outcome=correct]]")
    assert session.state.asked[0]["question"] != "tampered"


def test_finished_flag_and_result_guard():
    session = new_session(STRONG, max_questions=2)
    with pytest.raises(RuntimeError, match="not finished"):
        session.result()
    assert not session.finished
    session.submit("[[sim outcome=correct]]")
    assert not session.finished
    session.submit("[[sim outcome=correct]]")
    assert session.finished and session.next_prompt() is None
    result = session.result()
    assert result["stop_reason"] == "max_questions" and len(result["turns"]) == 2
    assert_valid(result)
    with pytest.raises(RuntimeError, match="over"):
        session.submit("late")


def test_result_is_json_and_does_not_share_state_with_the_session():
    session = new_session(EXAMPLE_CANDIDATE)
    drive(session, RuleBasedRespondent(EXAMPLE_CANDIDATE))
    result = session.result()
    assert json.loads(json.dumps(result)) == result
    result["turns"][0]["answer"] = "changed"
    assert session.result()["turns"][0]["answer"] != "changed"


def test_a_follow_up_waits_for_its_own_answer():
    candidate = make_candidate(3, misread_rate=1.0, id="CAND-MISREAD")
    session = new_session(candidate)
    main = session.next_prompt()
    session.submit("[[sim outcome=misread]]")
    follow_up = session.next_prompt()
    assert follow_up["kind"] == "follow_up" and follow_up["turn"] == main["turn"]
    assert len(session.turns) == 0 and session.current_exchange["answer"] == "[[sim outcome=misread]]"
    session.submit("[[sim outcome=correct]]")
    assert len(session.turns) == 1 and session.turns[0]["grade"]["score"] == 2
    assert session.next_prompt()["kind"] == "main"


# ---- views for the interviewer ---------------------------------------------------------------------

def test_scorecard_view_follows_the_scorecard():
    session = new_session(STRONG)
    view = session.scorecard_view()
    assert set(view) == {"py", "sql", "ds"}
    assert all(row["n"] == 0 and row["se"] == pytest.approx(1.0, abs=0.01) and not row["confident"]
               for row in view.values())
    assert view["py"]["minimum"] == job_for(STRONG)["min_levels"]["py"]
    session.submit("[[sim outcome=correct]]")
    after = session.scorecard_view()
    assert after["py"]["n"] == 1 and after["py"]["estimate"] > 0 and after["py"]["se"] < 1.0
    assert set(after["py"]) >= {"estimate", "se", "level", "n", "minimum", "confident"}
    drive(session, RuleBasedRespondent(STRONG))
    last = session.result()["turns"][-1]["scorecard_after"]
    for skill, row in session.scorecard_view().items():
        assert row["estimate"] == last[skill]["estimate"] and row["level"] == last[skill]["level"]


def test_interviewer_view_shows_the_open_diagnosis_and_misconceptions():
    candidate, job, source, _ = misconception_setup("py.variables-mutability.assignment-copies")
    session = new_session(candidate, job=job, components=make_components(source))
    assert session.interviewer_view()["current"] is None
    session.submit(RuleBasedRespondent(candidate).reply("", question=session.next_prompt()["question"],
                                                         kind="main", turn=1))
    view = session.interviewer_view()
    assert view["current"]["provisional_grade"]["score"] == 0
    assert view["current"]["diagnosis"]["cause"] == "misconception"
    assert [m["id"] for m in view["misconceptions"]] == ["py.variables-mutability.assignment-copies"]


# ---- failures ------------------------------------------------------------------------------------------

def test_a_failing_grader_finishes_the_session_with_an_error_result():
    class Broken:
        def grade(self, question, answer, follow_ups=()):
            raise RuntimeError("rate limited")

    session = new_session(STRONG, components=make_components(grader=Broken()))
    session.submit("[[sim outcome=correct]]")
    assert session.finished and session.next_prompt() is None
    result = session.result()
    assert result["stop_reason"] == "error" and result["turns"] == []
    assert any("rate limited" in e for e in result["errors"])
    assert_valid(result)


def test_run_interview_ends_with_an_error_result_when_the_respondent_fails():
    class Dies:
        def reply(self, prompt, *, question, kind, turn):
            if turn == 2:
                raise ConnectionError("browser closed")
            return "[[sim outcome=correct]]"

    result, _ = run(STRONG, "A2", respondent=Dies())
    assert result["stop_reason"] == "error" and len(result["turns"]) == 1
    assert any("browser closed" in e for e in result["errors"])
    assert_valid(result)


def test_session_needs_a_diagnoser_unless_diagnosis_is_off():
    components = make_components()
    components.diagnoser = None
    with pytest.raises(ValueError):
        new_session(STRONG, components=components)
    new_session(STRONG, "B1", components=components)


def test_empty_bank_session_finishes_at_once():
    from engine_support import FilteredSource
    session = new_session(STRONG, components=make_components(FilteredSource(lambda q: False)))
    assert session.finished and session.next_prompt() is None
    result = session.result()
    assert result["stop_reason"] == "bank_exhausted" and result["report"]["decision"] == "needs_review"
    assert_valid(result)
