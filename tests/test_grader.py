import pytest

from fake_llm import FakeLLM
from llm.client import DEFAULT_REASONING_EFFORT, LLMError, LLMOutputError
from part3.grader import GRADE_SCHEMA, PromptedGrader
from shared.interfaces import Grader

QUESTION = {
    "id": "SQL-L2-001", "skill": "sql", "level": 2, "type": "read-code", "topics": ["sql-joins"],
    "question": "What does this query return?",
    "code": "SELECT c.name FROM customers c LEFT JOIN orders o ON o.cid = c.id WHERE o.id IS NULL;",
    "tables": [{"name": "customers", "columns": ["id", "name"], "rows": [[1, "Ann"], [2, "Bo"], [3, None]]}],
    "model_answer": "Customers who have placed no orders.",
    "key_points": ["LEFT JOIN keeps unmatched customers", "IS NULL finds the unmatched"],
    "rubric": {"0": "Says it returns all customers", "1": "Partly right", "2": "Anti-join explained"},
    "misconceptions": ["sql.joins.left-join-drops"],
}
MISCONCEPTIONS = {"sql.joins.left-join-drops": {"id": "sql.joins.left-join-drops",
                                                 "description": "Believes LEFT JOIN drops unmatched rows"}}


def reply(score, rationale="ok"):
    return {"rationale": rationale, "score": score}


def make(replies, **kw):
    llm = FakeLLM(replies)
    return PromptedGrader(llm, "test-model", misconceptions=MISCONCEPTIONS, **kw), llm


def test_implements_grader_protocol():
    g, _ = make([reply(2)])
    assert isinstance(g, Grader)


def test_majority_vote_and_agreement():
    g, llm = make([reply(2, "a"), reply(1, "b"), reply(2, "c")])
    grade = g.grade(QUESTION, "answer")
    assert grade["score"] == 2 and grade["votes"] == [2, 1, 2]
    assert grade["agreement"] == pytest.approx(2 / 3)
    assert grade["grader"] == "test-model"
    assert grade["rationale"] == "a"
    assert len(llm.calls) == 3


def test_three_way_tie_uses_median():
    g, _ = make([reply(0), reply(2), reply(1)])
    grade = g.grade(QUESTION, "answer")
    assert grade["score"] == 1 and grade["votes"] == [0, 2, 1]
    assert grade["agreement"] == pytest.approx(1 / 3)


def test_two_way_tie_with_even_votes_takes_lower_median():
    g, _ = make([reply(0), reply(2)], votes=2)
    assert g.grade(QUESTION, "answer")["score"] == 0


def test_each_vote_has_its_own_cache_tag_and_settings_are_passed():
    g, llm = make([reply(2)], temperature=0.4, reasoning_effort="low")
    g.grade(QUESTION, "answer")
    assert [c["cache_tag"] for c in llm.calls] == ["vote-0", "vote-1", "vote-2"]
    assert all(c["temperature"] == 0.4 and c["reasoning_effort"] == "low" and c["model"] == "test-model"
               for c in llm.calls)
    assert llm.calls[0]["messages"] == llm.calls[1]["messages"]   # identical prompt, only the tag differs


def test_default_reasoning_effort_and_votes():
    g, llm = make([reply(1)])
    g.grade(QUESTION, "answer")
    assert llm.calls[0]["reasoning_effort"] == DEFAULT_REASONING_EFFORT
    assert len(llm.calls) == 3 and llm.calls[0]["temperature"] == 0.7


def test_schema_only_allows_marks_0_1_2():
    g, llm = make([reply(1)])
    g.grade(QUESTION, "answer")
    assert llm.calls[0]["json_schema"] == GRADE_SCHEMA
    assert GRADE_SCHEMA["properties"]["score"]["enum"] == [0, 1, 2]


def test_invalid_votes_are_dropped_but_others_count():
    g, _ = make([{"rationale": "x", "score": 7}, "not json", reply(1)])
    grade = g.grade(QUESTION, "answer")
    assert grade["votes"] == [1] and grade["score"] == 1 and grade["agreement"] == 1.0


def test_all_votes_invalid_raises():
    g, _ = make(["not json"])
    with pytest.raises(LLMOutputError, match="no usable grading vote"):
        g.grade(QUESTION, "answer")


def test_transport_errors_propagate():
    g, _ = make([LLMError("HTTP 500")])
    with pytest.raises(LLMError):
        g.grade(QUESTION, "answer")


def test_prompt_contains_everything_the_grader_needs():
    g, llm = make([reply(2)], votes=1)
    follow_ups = [{"prompt": "Which rows are kept?", "answer": "The ones with NULL order id.",
                   "diagnosis": {"cause": "misread"}}]
    g.grade(QUESTION, "It returns all customers.", follow_ups)
    system, user = (m["content"] for m in llm.calls[0]["messages"])
    assert "WHOLE exchange" in system and "misconception" in system
    for needle in ["What does this query return?", "LEFT JOIN orders", "Table customers",
                   "1 | Ann", "NULL", "Customers who have placed no orders.", "IS NULL finds the unmatched",
                   "Says it returns all customers", "Anti-join explained",
                   "sql.joins.left-join-drops: Believes LEFT JOIN drops unmatched rows",
                   "It returns all customers.", "Which rows are kept?", "The ones with NULL order id."]:
        assert needle in user, needle
    assert "diagnosis" not in user   # the grader is not primed by the diagnoser


def test_misconceptions_load_from_data_dir_by_default():
    llm = FakeLLM([reply(2)])
    g = PromptedGrader(llm, "m", votes=1)
    q = dict(QUESTION, misconceptions=["sql.joins.left-join-drops"])
    from part3.prompts import load_misconceptions
    ids = load_misconceptions()
    assert len(ids) > 20
    mid = next(iter(ids))
    q["misconceptions"] = [mid]
    g.grade(q, "x")
    assert ids[mid]["description"] in llm.calls[0]["messages"][1]["content"]


def test_votes_must_be_positive():
    with pytest.raises(ValueError):
        PromptedGrader(FakeLLM([reply(2)]), "m", votes=0)
