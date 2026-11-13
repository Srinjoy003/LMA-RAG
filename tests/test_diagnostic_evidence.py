"""Synthetic evidence cases; no human annotation or live accuracy claims."""

import json

import pytest
from jsonschema import Draft202012Validator

from fake_llm import FakeLLM
from llm.client import LLMOutputError
from part3.diagnoser import StructuredDiagnoser, build_schema
from shared.evidence import claim_span
from shared.interfaces import SYSTEMS
from engine.followups import Exchange, MisconceptionTracker
from engine_support import ROOT, assert_valid, misconception_setup, make_components, run
from test_diagnoser import A, B, QUESTION, RECORDS

TEXT = "I think assignment copies the list."
QUOTE = "assignment copies the list"


def vote(quote=QUOTE, cause="misconception", mid=A, contradicting=None):
    return dict(cause=cause, misconception_id=mid, follow_up="Can you trace a small example?",
                confidence_expressed="high", supporting_quote=quote,
                evidence_reason="Explicitly says that assignment creates a copy.",
                contradicting=contradicting or [])


def diagnose(votes, text=TEXT, **kwargs):
    llm = FakeLLM(votes)
    d = StructuredDiagnoser(llm, "test-model", RECORDS, evidence_mode=True, **kwargs)
    return d.diagnose(QUESTION, text), llm


def test_quote_offsets_are_exact_unicode_character_offsets():
    text = "λ: " + TEXT
    span = claim_span(dict(misconception_id=A, supporting_quote=QUOTE, reason="copy claim"), text, [A])
    assert text[span["quote_start"]:span["quote_end"]] == QUOTE


@pytest.mark.parametrize("quote", [None, "", "   ", "invented quote", "Both names refer to the same list"])
def test_unsupported_or_reference_only_quote_cannot_support_diagnosis(quote):
    result, _ = diagnose([vote(quote)])
    assert result["cause"] == "gap" and result["misconception_id"] is None
    assert result["follow_up"] is None


def test_verified_winning_quote_and_version_are_returned():
    result, llm = diagnose([vote(), vote(), vote(cause="gap", mid=None)])
    assert result["misconception_id"] == A and result["supporting_quote"] == QUOTE
    assert result["vote_share"] == pytest.approx(2 / 3)
    assert result["prompt_version"] == "evidence-v1"
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "O(n)" in prompt and "O(1)" in prompt and "CURRENT" in prompt


def test_plurality_or_one_usable_vote_cannot_confirm_a_belief():
    result, _ = diagnose([vote(), vote("list", mid=B), vote(cause="gap", mid=None)])
    assert result["misconception_id"] is None
    result, _ = diagnose([vote(), LLMOutputError("bad"), LLMOutputError("bad")])
    assert result["misconception_id"] is None


def test_contradictions_require_majority_of_distinct_votes_not_duplicate_claims():
    text = "Assignment does not copy the list."
    claim = dict(misconception_id=A, supporting_quote=text, reason="Explicit rejection of copying.")
    result, _ = diagnose([vote(None, "correct", None, [claim] * 4),
                          vote(None, "correct", None), vote(None, "correct", None)], text)
    assert result["contradicting"] == []
    result, _ = diagnose([vote(None, "correct", None, [claim]),
                          vote(None, "correct", None, [claim]), vote(None, "correct", None)], text)
    assert len(result["contradicting"]) == 1
    assert result["contradicting"][0]["vote_share"] == pytest.approx(2 / 3)


def test_empty_catalog_produces_valid_evidence_schema():
    Draft202012Validator.check_schema(build_schema([], True))


def test_live_factory_enables_evidence_only_for_new_preset(monkeypatch):
    from engine.factory import build_components
    monkeypatch.setattr("llm.client.get_client", lambda: FakeLLM([vote()]))
    assert build_components(SYSTEMS["A2E"], live=True).diagnoser.evidence_mode
    assert not build_components(SYSTEMS["A2"], live=True).diagnoser.evidence_mode
    assert SYSTEMS["A2E"].to_dict()["diagnosis_prompt_version"] == "evidence-v1"
    assert "diagnosis_prompt_version" not in SYSTEMS["A2"].to_dict()


def test_conflicting_support_and_contradiction_cannot_add_positive_evidence():
    claim = dict(misconception_id=A, supporting_quote=QUOTE, reason="Contradiction asserted by model.")
    result, _ = diagnose([vote(contradicting=[claim])])
    assert result["misconception_id"] is None


class FixedDiagnoser:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def diagnose(self, *args):
        self.calls += 1
        return self.result


def exchange(tracker, diagnosis, answer=TEXT, score=0, turn=1):
    return Exchange(QUESTION, QUESTION["question"], answer, turn=turn,
                    diagnoser=diagnosis, config=SYSTEMS["A2E"], tracker=tracker,
                    grade=lambda *args: {"score": score, "votes": [score] * 3}, known_ids=set(RECORDS))


def test_engine_rejects_custom_diagnoser_with_fabricated_quote():
    tracker = MisconceptionTracker(unit="question")
    e = exchange(tracker, FixedDiagnoser(dict(vote("fabricated"), vote_share=1)))
    assert tracker.findings() == [] and e.pending is None
    assert "evidence_rejected" in e.diagnosis


def test_correct_reply_can_dispute_prior_confirmation_without_asking_followup():
    tracker = MisconceptionTracker(unit="question")
    for turn, qid in [(1, "Q1"), (2, "Q2")]:
        tracker.record(A, turn, question_id=qid, reply_text=TEXT)
    assert tracker.is_confirmed(A)
    text = "Assignment does not copy the list."
    claim = dict(misconception_id=A, supporting_quote=text, reason="Explicit rejection of copying.", vote_share=2/3)
    d = FixedDiagnoser(dict(vote(None, "correct", None, [claim]), vote_share=1))
    e = exchange(tracker, d, text, score=2, turn=3)
    assert d.calls == 1 and e.pending is None
    finding = tracker.findings()[0]
    assert not finding["confirmed"] and finding["status"] == "disputed"
    saved = json.loads((ROOT / "examples/interview_result.example.json").read_text())
    saved["report"]["misconceptions"] = [finding]
    assert_valid(saved)
    assert finding["contradicting"][0]["vote_share"] == pytest.approx(2/3)
    proof = finding["contradicting"][0]
    assert text[proof["quote_start"]:proof["quote_end"]] == proof["supporting_quote"]
    tracker.record(A, 4, question_id="Q4", reply_text=TEXT)
    assert not tracker.is_confirmed(A)  # unresolved contradictions are not silently erased
    finding["contradicting"][0]["reason"] = "changed snapshot"
    assert tracker.findings()[0]["contradicting"][0]["reason"] != "changed snapshot"


def test_rejecting_never_attributed_belief_does_not_create_a_finding():
    tracker = MisconceptionTracker(unit="question")
    text = "Assignment does not copy the list."
    claim = dict(misconception_id=A, supporting_quote=text, reason="Rejects copying.")
    exchange(tracker, FixedDiagnoser(dict(vote(None, "correct", None, [claim]), vote_share=1)), text, 2)
    assert tracker.findings() == []


def test_engine_saves_supporting_quote_offsets_and_reason():
    tracker = MisconceptionTracker(unit="question")
    exchange(tracker, FixedDiagnoser(dict(vote(), vote_share=1)))
    evidence = tracker.findings()[0]["evidence"][0]
    assert evidence["supporting_quote"] == QUOTE and evidence["reason"]
    assert TEXT[evidence["quote_start"]:evidence["quote_end"]] == QUOTE


def test_factory_stub_and_agent_use_evidence_preset_and_old_results_still_validate():
    from engine.factory import build_components
    from test_engine_agent import ScriptedLLM, agent_run
    candidate, job, source, _ = misconception_setup(A)
    components = build_components(SYSTEMS["A2E"], live=False)
    result, _ = run(candidate, "A2E", job=job, components=components)
    assert result["errors"] == []
    assert any(m["evidence"][0].get("supporting_quote") for m in result["report"]["misconceptions"])
    assert_valid(result)
    agent_result = agent_run(candidate, ScriptedLLM(), "A2E", job=job,
                             components=build_components(SYSTEMS["A2E"], live=False))
    assert agent_result["errors"] == []
    assert agent_result["report"]["misconceptions"] == result["report"]["misconceptions"]
    assert_valid(agent_result)
    assert_valid(json.loads((ROOT / "examples/interview_result.example.json").read_text()))
    assert not SYSTEMS["A2Q"].diagnostic_evidence and not SYSTEMS["A4"].diagnostic_evidence
