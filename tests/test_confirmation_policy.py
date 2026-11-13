"""Independent-question corroboration, historical controls, and result provenance."""

import pytest

from engine.followups import MisconceptionTracker
from shared.interfaces import SYSTEMS
from engine_support import FilteredSource, assert_valid, make_components, misconception_setup, run

MID = "py.variables-mutability.assignment-copies"


def test_same_question_cannot_confirm_even_when_repeated_on_another_turn():
    tracker = MisconceptionTracker(2, unit="question")
    for turn, index in [(1, 0), (1, 1), (1, 2), (4, 0)]:
        tracker.record(MID, turn, 0.667, question_id="Q1", reply_index=index,
                       reply_text="A second variable gets a copy.")
    assert tracker.count(MID) == 1
    assert not tracker.is_confirmed(MID)
    tracker.record(MID, 5, question_id="Q2", reply_text="Assignment copies it.")
    assert tracker.is_confirmed(MID)
    finding = tracker.findings()[0]
    assert finding["confirmation_count"] == 2
    assert len(finding["evidence"]) == 5
    assert finding["evidence_turns"] == [1, 4, 5]
    # Callers cannot mutate the tracker's records through a report snapshot.
    finding["evidence"][0]["question_id"] = "changed"
    assert tracker.findings()[0]["evidence"][0]["question_id"] == "Q1"


def test_question_identity_is_required_without_partial_mutation():
    tracker = MisconceptionTracker(unit="question")
    with pytest.raises(ValueError, match="question_id"):
        tracker.record(MID, 1)
    assert tracker.findings() == []


@pytest.mark.parametrize("kwargs", [{"unit": "turn"}, {"confirmations": 0},
                                   {"confirmations": True}, {"confirmations": 1.5}])
def test_invalid_confirmation_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MisconceptionTracker(**kwargs)


@pytest.mark.parametrize("system,confirmed", [("A2", True), ("A2Q", False), ("A4", True)])
def test_single_question_interview_distinguishes_policies_and_preserves_a4(system, confirmed):
    candidate, job, _, topic = misconception_setup(MID)
    source = FilteredSource(lambda q: topic not in q["topics"] or q["id"] == "PY-L2-006")
    result, _ = run(candidate, system, job=job, components=make_components(source))
    assert result["errors"] == []
    finding = next(m for m in result["report"]["misconceptions"] if m["id"] == MID)
    assert finding["confirmed"] is confirmed
    assert {e["question_id"] for e in finding["evidence"]} == {"PY-L2-006"}
    turns = {t["index"]: t for t in result["turns"]}
    for e in finding["evidence"]:
        turn = turns[e["turn"]]
        actual = turn["answer"] if e["reply_index"] == 0 else turn["follow_ups"][e["reply_index"] - 1]["answer"]
        assert e["reply_text"] == actual
    if system == "A2Q":
        assert finding["confirmation_count"] == 1
        assert "insufficient corroboration" in result["report"]["summary"]
    assert_valid(result)


def test_revised_system_can_confirm_on_a_second_question():
    candidate, job, source, _ = misconception_setup(MID)
    result, _ = run(candidate, "A2Q", job=job, components=make_components(source))
    assert result["errors"] == []
    finding = next(m for m in result["report"]["misconceptions"] if m["id"] == MID)
    assert finding["confirmed"]
    assert finding["confirmation_count"] >= 2
    assert len({e["question_id"] for e in finding["evidence"]}) >= 2
    assert_valid(result)


def test_agent_and_code_driver_apply_same_question_policy():
    from test_engine_agent import ScriptedLLM, agent_run
    candidate, job, source, _ = misconception_setup(MID)
    result = agent_run(candidate, ScriptedLLM(), "A2Q", job=job,
                       components=make_components(source))
    code, _ = run(candidate, "A2Q", job=job, components=make_components(source))
    assert result["errors"] == []
    assert result["report"]["misconceptions"] == code["report"]["misconceptions"]
    assert result["exchanges"] == code["exchanges"]
    assert_valid(result)


def test_experiment_names_and_saved_configuration_keep_old_and_new_apart():
    assert SYSTEMS["A2"].misconception_confirmation_unit == "reply"
    assert SYSTEMS["A4"].misconception_confirmation_unit == "reply"
    assert SYSTEMS["A4"].misconception_confirmations == 1
    assert SYSTEMS["A2Q"].to_dict()["misconception_confirmation_unit"] == "question"
    assert SYSTEMS["A2Q"].system != SYSTEMS["A2"].system
