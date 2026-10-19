import pytest

from fake_llm import FakeLLM
from llm.client import LLMOutputError
from part3.diagnoser import StructuredDiagnoser, build_schema
from shared.interfaces import Diagnoser

A, B = "py.variables-mutability.assignment-copies", "py.variables-mutability.string-method-mutates"
OTHER = "py.control-flow.range-inclusive"   # a real id, but not on this question
QUESTION = {
    "id": "PY-L1-002", "skill": "py", "level": 1, "type": "explain", "topics": ["py-variables-mutability"],
    "question": "What happens to a when we run b = a; b.append(4)?",
    "model_answer": "Both names refer to the same list, so a also shows the 4.",
    "key_points": ["b is another name for the same list", "a sees the change"],
    "rubric": {"0": "x", "1": "y", "2": "z"},
    "misconceptions": [A, B],
}
RECORDS = {A: {"id": A, "description": "Believes assignment copies a mutable object"},
           B: {"id": B, "description": "Believes string methods mutate in place"}}


def vote(cause="misconception", mid=A, follow_up="Trace a after b.append(4)?", conf="high"):
    return {"cause": cause, "misconception_id": mid, "follow_up": follow_up, "confidence_expressed": conf}


def make(replies, **kw):
    llm = FakeLLM(replies)
    return StructuredDiagnoser(llm, "test-model", RECORDS, **kw), llm


def test_implements_diagnoser_protocol():
    d, _ = make([vote()])
    assert isinstance(d, Diagnoser)


def test_majority_over_cause_and_id():
    d, llm = make([vote(), vote("gap", None, None, "low"), vote(follow_up="Show me a after the append.")])
    diag = d.diagnose(QUESTION, "b is its own copy")
    assert diag["cause"] == "misconception" and diag["misconception_id"] == A
    assert diag["vote_share"] == pytest.approx(2 / 3)
    assert len(diag["votes"]) == 3 and diag["votes"][1]["cause"] == "gap"
    assert diag["follow_up"] == "Trace a after b.append(4)?"     # first winning vote
    assert diag["confidence_expressed"] == "high"
    assert [c["cache_tag"] for c in llm.calls] == ["vote-0", "vote-1", "vote-2"]


def test_follow_up_comes_from_a_winning_vote_not_a_losing_one():
    d, _ = make([vote("gap", None, "Losing question?"), vote(follow_up=None), vote(follow_up="Winning question?")])
    diag = d.diagnose(QUESTION, "x")
    assert diag["cause"] == "misconception" and diag["follow_up"] == "Winning question?"


def test_same_cause_different_id_is_not_agreement():
    d, _ = make([vote(mid=A), vote(mid=B), vote(mid=A)])
    diag = d.diagnose(QUESTION, "x")
    assert diag["misconception_id"] == A and diag["vote_share"] == pytest.approx(2 / 3)


def test_id_not_on_the_question_list_is_dropped():
    # The schema forbids it, so the fake rejects it as a bad vote; a lenient provider is
    # handled by _clean as well (tested directly).
    d, _ = make([vote(mid=A), vote(mid=A), vote(mid=A)])
    assert d.diagnose(QUESTION, "x")["misconception_id"] == A
    cleaned = StructuredDiagnoser._clean(vote(mid=OTHER), QUESTION["misconceptions"])
    assert cleaned["misconception_id"] is None and cleaned["cause"] == "misconception"


def test_schema_rejects_unlisted_ids_and_bad_values():
    d, llm = make([vote(mid=OTHER), vote("blunder"), vote(mid=B)])
    diag = d.diagnose(QUESTION, "x")
    assert diag["votes"] == [{"cause": "misconception", "misconception_id": B,
                              "follow_up": "Trace a after b.append(4)?", "confidence_expressed": "high"}]
    assert diag["vote_share"] == 1.0
    schema = llm.calls[0]["json_schema"]
    assert schema == build_schema([A, B])
    assert schema["properties"]["misconception_id"]["enum"] == [A, B, None]
    assert schema["properties"]["cause"]["enum"] == ["gap", "misconception", "misread", "slip"]
    assert schema["properties"]["confidence_expressed"]["enum"] == ["low", "medium", "high"]


def test_id_is_cleared_unless_cause_is_misconception():
    d, _ = make([vote("slip", A, "Look again?")] * 3)
    diag = d.diagnose(QUESTION, "x")
    assert diag["cause"] == "slip" and diag["misconception_id"] is None


def test_tie_goes_to_the_first_vote():
    d, _ = make([vote("misread", None, "What does the question ask?"), vote("gap", None, None),
                 vote("slip", None, "Check line 2?")])
    diag = d.diagnose(QUESTION, "x")
    assert diag["cause"] == "misread" and diag["vote_share"] == pytest.approx(1 / 3)


def test_null_and_blank_follow_up_become_none():
    d, _ = make([vote("gap", None, "  ", "low")] * 3)
    diag = d.diagnose(QUESTION, "no idea")
    assert diag["follow_up"] is None and diag["confidence_expressed"] == "low"


def test_all_invalid_votes_raise_and_partial_failures_are_tolerated():
    d, _ = make(["nope"])
    with pytest.raises(LLMOutputError, match="no usable diagnosis vote"):
        d.diagnose(QUESTION, "x")
    d, _ = make(["nope", vote(), vote()])
    diag = d.diagnose(QUESTION, "x")
    assert len(diag["votes"]) == 2 and diag["vote_share"] == 1.0


def test_prompt_has_question_descriptions_answer_and_history():
    d, llm = make([vote()], votes=1)
    history = [{"prompt": QUESTION["question"] + "\n\nextra", "answer": "b is a fresh copy."},
               {"prompt": "What does b.append do?", "answer": "It changes only b."}]
    d.diagnose(QUESTION, "Then a is unchanged.", history)
    system, user = (m["content"] for m in llm.calls[0]["messages"])
    assert "gap" in system and "misread" in system and "slip" in system
    for needle in [QUESTION["question"], A + ": Believes assignment copies", B + ": Believes string methods",
                   "Both names refer to the same list", "b is a fresh copy.", "What does b.append do?",
                   "It changes only b.", "Then a is unchanged.", "(the question above)"]:
        assert needle in user, needle
    assert user.rindex("Then a is unchanged.") > user.rindex("It changes only b.")   # answer comes last


def test_no_history_section_when_history_is_empty():
    d, llm = make([vote()], votes=1)
    d.diagnose(QUESTION, "x")
    assert "EARLIER EXCHANGES" not in llm.calls[0]["messages"][1]["content"]


def test_default_misconceptions_load_from_data_dir():
    d = StructuredDiagnoser(FakeLLM([vote()]), "m", votes=1)
    assert A in d.misconceptions
