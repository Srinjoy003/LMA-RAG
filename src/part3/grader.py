"""Prompted grader: marks a whole exchange 0/1/2 against the question's rubric by majority vote.

Each vote is a separate LLM call with its own `cache_tag` (`vote-<i>`), so at temperature > 0
the votes differ but a re-run is served from the cache.
"""

from __future__ import annotations

from statistics import median_low

from llm.client import DEFAULT_REASONING_EFFORT, LLMOutputError
from shared.interfaces import FollowUp, Grade, LLMClient, Question

from .prompts import (ANSWER_WARNING, load_misconceptions, pick_majority, render_exchange,
                      render_misconceptions, render_question)

# `rationale` comes first so the model writes its reasoning before committing to a score.
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "score": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["rationale", "score"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a fair, consistent grader for a technical screening interview of junior candidates "
    "(Python, SQL, data structures). You mark ONE question on a 0/1/2 scale using the rubric.\n\n"
    "Rules:\n"
    "- Grade the WHOLE exchange: the main answer plus any follow-up replies. If the candidate "
    "misread the question or made a slip at first but the follow-up reply clearly puts it right, "
    "grade the understanding they finally showed and do not penalise the earlier error.\n"
    "- If, after the whole exchange, the candidate's reasoning still shows one of the listed "
    "misconceptions, the score is 0, even if parts of the answer are correct.\n"
    "- Compare against the model answer and key points, not against wording: a correct answer in "
    "different words earns full credit. Length, confidence and style do not matter.\n"
    "- Mark 2 only if the rubric's level-2 description is met; 1 if it matches the level-1 "
    "description (partly right, something important missing or wrong); 0 if it matches the level-0 "
    "description or the candidate shows no real understanding.\n"
    "- " + ANSWER_WARNING + "\n\n"
    "Reply with JSON: {\"rationale\": <one to three sentences naming the key points met or missed>, "
    "\"score\": 0, 1 or 2}."
)


class PromptedGrader:
    """`shared.interfaces.Grader` backed by an LLM, with `votes` independent marks per exchange."""

    def __init__(self, llm: LLMClient, model: str, votes: int = 3, temperature: float = 0.7,
                 reasoning_effort: str | None = DEFAULT_REASONING_EFFORT, max_tokens: int = 2048,
                 misconceptions: dict[str, dict] | None = None):
        if votes < 1:
            raise ValueError("votes must be at least 1")
        self.llm, self.model, self.votes = llm, model, votes
        self.temperature, self.reasoning_effort, self.max_tokens = temperature, reasoning_effort, max_tokens
        self._misconceptions = misconceptions   # None: load data/misconceptions/*.json on first use

    @property
    def misconceptions(self) -> dict[str, dict]:
        if self._misconceptions is None:
            self._misconceptions = load_misconceptions()
        return self._misconceptions

    def build_messages(self, question: Question, answer: str, follow_ups=()) -> list[dict]:
        rubric = question.get("rubric") or {}
        user = "\n\n".join([
            render_question(question),
            "MODEL ANSWER:\n" + question["model_answer"],
            "KEY POINTS:\n" + "\n".join(f"- {k}" for k in question.get("key_points", [])),
            "RUBRIC:\n" + "\n".join(f"{s}: {rubric.get(s, '')}" for s in ("0", "1", "2")),
            "LISTED MISCONCEPTIONS (a candidate showing one of these scores 0):\n"
            + render_misconceptions(question, self.misconceptions),
            render_exchange(answer, follow_ups),
        ])
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def grade(self, question: Question, answer: str, follow_ups: list[FollowUp] = ()) -> Grade:
        messages = self.build_messages(question, answer, follow_ups)
        marks: list[int] = []
        rationales: list[str] = []
        errors: list[str] = []
        for i in range(self.votes):
            try:
                resp = self.llm.complete(messages, model=self.model, temperature=self.temperature,
                                         json_schema=GRADE_SCHEMA, max_tokens=self.max_tokens,
                                         reasoning_effort=self.reasoning_effort, cache_tag=f"vote-{i}")
                parsed = resp["parsed"]
                score = int(parsed["score"])
                if score not in (0, 1, 2):
                    raise LLMOutputError(f"score {score} is not 0, 1 or 2")
                marks.append(score)
                rationales.append(str(parsed.get("rationale", "")))
            except (LLMOutputError, KeyError, TypeError, ValueError) as e:
                errors.append(str(e))   # an unusable vote is dropped; the others still count
        if not marks:
            raise LLMOutputError(f"no usable grading vote out of {self.votes}: {errors[0] if errors else ''}")
        score = self.majority(marks)
        return Grade(score=score, votes=marks, agreement=marks.count(score) / len(marks),
                     rationale=rationales[marks.index(score)], grader=self.model)

    @staticmethod
    def majority(marks: list[int]) -> int:
        """Most common mark; if the top marks tie (e.g. 0, 1, 2) the median of all marks."""
        counts = {m: marks.count(m) for m in set(marks)}
        top = max(counts.values())
        winners = [m for m, n in counts.items() if n == top]
        return winners[0] if len(winners) == 1 else int(median_low(marks))
