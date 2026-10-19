"""Structured diagnoser: why was an answer wrong (cause + misconception id) and what to ask next.

Runs `votes` independent LLM calls (own `cache_tag` each) and takes a majority over
(cause, misconception_id). The diagnoser only suggests; the engine decides what to confirm.
"""

from __future__ import annotations

from llm.client import DEFAULT_REASONING_EFFORT, LLMOutputError
from shared.interfaces import Diagnosis, FollowUp, LLMClient, Question

from .prompts import (ANSWER_WARNING, pick_majority, render_history, render_misconceptions,
                      render_question)

CAUSES = ("gap", "misconception", "misread", "slip")
CONFIDENCE = ("low", "medium", "high")

SYSTEM = (
    "You are the diagnostic part of an adaptive technical interviewer for junior candidates "
    "(Python, SQL, data structures). A candidate's answer was wrong, partial or vague. Decide WHY, "
    "choosing exactly one cause:\n"
    "- gap: the candidate does not know the concept, or the answer is empty, vague or off-topic "
    "without any specific wrong belief.\n"
    "- misconception: the answer shows a specific, confidently held wrong belief about how things "
    "work. If it matches one of the question's listed misconceptions, give its id; use null if it "
    "matches none of them.\n"
    "- misread: the candidate seems to know the topic but misunderstood what the question or the "
    "code/tables asked (answered a different question, misread a line of code, mixed up a name).\n"
    "- slip: the idea is right but a careless error crept in (typo, off-by-one in the explanation, "
    "swapped two words), the kind of thing they would fix themselves if asked to look again.\n\n"
    "Also give `follow_up`: ONE short, neutral question to ask the candidate next that would best "
    "tell the likely causes apart (for example ask them to trace a tiny concrete case, or to restate "
    "what the question asks). Do not reveal the correct answer, do not name the misconception, and do "
    "not repeat a follow-up that was already asked. Use null if the cause is clearly gap (nothing "
    "useful to probe) or the answer is already correct.\n"
    "And give `confidence_expressed`: how sure the candidate SOUNDED (low = hedging or guessing, "
    "high = assertive), whether or not they were right.\n\n"
    + ANSWER_WARNING + "\n\n"
    "Reply with JSON: {\"cause\", \"misconception_id\", \"follow_up\", \"confidence_expressed\"}. "
    "misconception_id must be null unless cause is misconception."
)


def build_schema(misconception_ids: list[str]) -> dict:
    ids: list = list(dict.fromkeys(misconception_ids))
    return {
        "type": "object",
        "properties": {
            "cause": {"type": "string", "enum": list(CAUSES)},
            "misconception_id": {"type": ["string", "null"], "enum": ids + [None]},
            "follow_up": {"type": ["string", "null"]},
            "confidence_expressed": {"type": "string", "enum": list(CONFIDENCE)},
        },
        "required": ["cause", "misconception_id", "follow_up", "confidence_expressed"],
        "additionalProperties": False,
    }


class StructuredDiagnoser:
    """`shared.interfaces.Diagnoser` backed by an LLM, majority vote over `votes` runs."""

    def __init__(self, llm: LLMClient, model: str, misconceptions: dict[str, dict] | None = None,
                 votes: int = 3, temperature: float = 0.7,
                 reasoning_effort: str | None = DEFAULT_REASONING_EFFORT, max_tokens: int = 2048):
        if votes < 1:
            raise ValueError("votes must be at least 1")
        if misconceptions is None:
            from .prompts import load_misconceptions
            misconceptions = load_misconceptions()
        self.llm, self.model, self.misconceptions, self.votes = llm, model, misconceptions, votes
        self.temperature, self.reasoning_effort, self.max_tokens = temperature, reasoning_effort, max_tokens

    def build_messages(self, question: Question, answer: str, history=()) -> list[dict]:
        parts = [
            render_question(question),
            "MODEL ANSWER (for your reference only):\n" + question["model_answer"],
            "KEY POINTS:\n" + "\n".join(f"- {k}" for k in question.get("key_points", [])),
            "LISTED MISCONCEPTIONS FOR THIS QUESTION:\n" + render_misconceptions(question, self.misconceptions),
        ]
        if history:
            parts.append("EARLIER EXCHANGES ON THIS QUESTION (the answer to diagnose comes after them):\n"
                         + render_history(question, history))
        parts.append("ANSWER TO DIAGNOSE:\n<<<\n" + answer.strip() + "\n>>>")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def diagnose(self, question: Question, answer: str, history: list[FollowUp] = ()) -> Diagnosis:
        allowed = list(question.get("misconceptions") or [])
        schema = build_schema(allowed)
        messages = self.build_messages(question, answer, history)
        votes: list[dict] = []
        errors: list[str] = []
        for i in range(self.votes):
            try:
                resp = self.llm.complete(messages, model=self.model, temperature=self.temperature,
                                         json_schema=schema, max_tokens=self.max_tokens,
                                         reasoning_effort=self.reasoning_effort, cache_tag=f"vote-{i}")
                votes.append(self._clean(resp["parsed"], allowed))
            except (LLMOutputError, KeyError, TypeError, ValueError) as e:
                errors.append(str(e))
        if not votes:
            raise LLMOutputError(f"no usable diagnosis vote out of {self.votes}: {errors[0] if errors else ''}")

        keys = [(v["cause"], v["misconception_id"]) for v in votes]
        (cause, mid), n = pick_majority(keys)
        winners = [v for v in votes if (v["cause"], v["misconception_id"]) == (cause, mid)]
        follow_up = next((v["follow_up"] for v in winners if v["follow_up"]), None)
        confidence = pick_majority([v["confidence_expressed"] for v in winners])[0]
        return Diagnosis(cause=cause, misconception_id=mid, follow_up=follow_up,
                         vote_share=n / len(votes), votes=votes, confidence_expressed=confidence)

    @staticmethod
    def _clean(parsed: dict, allowed: list[str]) -> dict:
        """One vote in canonical form; an id that is not on the question's list is dropped."""
        cause = parsed.get("cause")
        if cause not in CAUSES:
            raise LLMOutputError(f"unknown cause {cause!r}")
        mid = parsed.get("misconception_id")
        if mid not in allowed or cause != "misconception":
            mid = None
        follow_up = parsed.get("follow_up")
        follow_up = follow_up.strip() if isinstance(follow_up, str) and follow_up.strip() else None
        conf = parsed.get("confidence_expressed")
        return {"cause": cause, "misconception_id": mid, "follow_up": follow_up,
                "confidence_expressed": conf if conf in CONFIDENCE else "medium"}
