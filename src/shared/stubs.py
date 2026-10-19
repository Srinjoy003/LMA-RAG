"""Fakes for every interface, so each part can run end to end before the others exist.

The rule-based respondent writes a hidden tag into each reply, e.g.
`[[sim outcome=misconception id=py.collections.tuple-mutable]]`. The tag graders and
diagnosers read it back, so a full interview runs with no LLM, deterministically.
Never use tags outside tests and dry runs.
"""

from __future__ import annotations

import random
import re

from .interfaces import (AnswerPlan, Candidate, Diagnosis, FollowUp, Grade, Question,
                         TurnKind)

EXPECTED_SCORE = {"correct": 2, "partial": 1, "slip": 1, "wrong": 0, "misconception": 0, "misread": 0}
_TAG = re.compile(r"\[\[sim outcome=(?P<outcome>[a-z]+)(?: id=(?P<id>[a-z0-9.\-]+))?\]\]")


# --------------------------------------------------------------------------
# Answer policy (reference version of interfaces.plan_answer; Part 4 owns the final rules)
# --------------------------------------------------------------------------

def effective_level(candidate: Candidate, question: Question) -> int:
    """Candidate's level on a question = their weakest level among its topics."""
    levels = candidate["topic_levels"]
    return min(levels.get(t, 0) for t in question["topics"])


def reference_plan_answer(candidate: Candidate, question: Question, *, kind: TurnKind, turn: int,
                          previous: AnswerPlan | None, rng: random.Random) -> AnswerPlan:
    """Main question:
        1. holds one of the question's misconceptions -> answers with it (score 0)
        2. misreads with probability misread_rate      -> misread (0)
        3. otherwise by gap = effective level - question level:
           gap >= 0 -> correct (2), or slip (1) with probability slip_rate
           gap == -1 -> partial (1) or wrong (0), 50/50
           gap <= -2 -> wrong (0)
    Follow-up (based on the main plan):
        misconception stays misconception; misread and slip are corrected once clarified
        (answered by the gap rule without misreads/slips); wrong and partial stay as they are.
    """
    level = effective_level(candidate, question)
    gap = level - question["level"]
    attack = None
    if kind == "main" and candidate.get("safety_attack") and turn in candidate["safety_attack"]["on_turns"]:
        attack = candidate["safety_attack"]["text"]

    def by_gap(allow_slip: bool) -> str:
        if gap >= 0:
            return "slip" if allow_slip and rng.random() < candidate.get("slip_rate", 0.05) else "correct"
        if gap == -1:
            return "partial" if rng.random() < 0.5 else "wrong"
        return "wrong"

    held = [m for m in question.get("misconceptions", []) if m in candidate.get("misconceptions", [])]
    misconception_id = None
    if kind == "main" or previous is None:
        if held:
            outcome, misconception_id = "misconception", held[0]
        elif rng.random() < candidate.get("misread_rate", 0.0):
            outcome = "misread"
        else:
            outcome = by_gap(allow_slip=True)
    else:
        prev = previous["outcome"]
        if prev == "misconception":
            outcome, misconception_id = "misconception", previous.get("misconception_id")
        elif prev in ("misread", "slip"):
            outcome = by_gap(allow_slip=False)
        else:
            outcome = prev

    return AnswerPlan(outcome=outcome, misconception_id=misconception_id,
                      expected_score=EXPECTED_SCORE[outcome], effective_level=level, attack=attack)


# --------------------------------------------------------------------------
# Respondents
# --------------------------------------------------------------------------

class RuleBasedRespondent:
    """Simulated candidate without an LLM: replies are a template plus a hidden tag.

    `truths` records (turn, kind, plan) for every reply, for the experiment runner to
    merge into the result. The engine must not read it.
    """

    def __init__(self, candidate: Candidate, seed: int | None = None):
        self.candidate = candidate
        default_seed = int(re.sub(r"\D", "", candidate["id"]) or 0)
        self.rng = random.Random(candidate.get("seed", default_seed) if seed is None else seed)
        self.truths: list[tuple[int, TurnKind, AnswerPlan]] = []
        self._last_main: AnswerPlan | None = None

    def reply(self, prompt: str, *, question: Question | None, kind: TurnKind, turn: int) -> str:
        if question is None:
            return "I'm not sure what you mean."
        previous = self._last_main if kind == "follow_up" else None
        plan = reference_plan_answer(self.candidate, question, kind=kind, turn=turn,
                                     previous=previous, rng=self.rng)
        if kind == "main":
            self._last_main = plan
        self.truths.append((turn, kind, plan))
        tag = f"[[sim outcome={plan['outcome']}" + (f" id={plan['misconception_id']}" if plan.get("misconception_id") else "") + "]]"
        text = f"My answer to {question['id']}. {tag}"
        if plan.get("attack"):
            text += " " + plan["attack"]
        return text


class ScriptedRespondent:
    """Replays fixed answers in order (useful for unit tests of the engine)."""

    def __init__(self, answers: list[str]):
        self.answers = list(answers)

    def reply(self, prompt: str, *, question: Question | None, kind: TurnKind, turn: int) -> str:
        return self.answers.pop(0) if self.answers else ""


# --------------------------------------------------------------------------
# Graders
# --------------------------------------------------------------------------

def read_tag(text: str) -> tuple[str | None, str | None]:
    """Last simulator tag in a text, as (outcome, misconception_id)."""
    matches = list(_TAG.finditer(text or ""))
    if not matches:
        return None, None
    return matches[-1]["outcome"], matches[-1]["id"]


class TagGrader:
    """Reads the simulator tag; with `noise`, each vote is replaced by a random mark."""

    def __init__(self, votes: int = 3, noise: float = 0.0, seed: int = 0):
        self.votes, self.noise, self.rng = votes, noise, random.Random(seed)

    def grade(self, question: Question, answer: str, follow_ups: list[FollowUp] = ()) -> Grade:
        texts = [answer] + [f.get("answer", "") for f in follow_ups]
        outcome = None
        for text in texts:  # the latest tagged reply decides (follow-ups can correct a misread)
            o, _ = read_tag(text)
            outcome = o or outcome
        true_score = EXPECTED_SCORE.get(outcome, 0)
        votes = [self.rng.choice([0, 1, 2]) if self.rng.random() < self.noise else true_score
                 for _ in range(self.votes)]
        score = max(set(votes), key=lambda v: (votes.count(v), -abs(v - 1)))
        return Grade(score=score, votes=votes, agreement=votes.count(score) / len(votes),
                     rationale=f"stub: tag outcome={outcome}", grader="tag-stub")


class RandomGrader:
    """Random marks: for testing the loop's plumbing only."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def grade(self, question: Question, answer: str, follow_ups: list[FollowUp] = ()) -> Grade:
        votes = [self.rng.choice([0, 1, 2]) for _ in range(3)]
        score = sorted(votes)[1]
        return Grade(score=score, votes=votes, agreement=votes.count(score) / 3, grader="random-stub")


# --------------------------------------------------------------------------
# Diagnosers
# --------------------------------------------------------------------------

_CAUSE = {"misconception": "misconception", "misread": "misread", "slip": "slip",
          "wrong": "gap", "partial": "gap"}


class TagDiagnoser:
    """Reads the simulator tag and asks a generic follow-up."""

    def diagnose(self, question: Question, answer: str, history: list[FollowUp] = ()) -> Diagnosis:
        outcome, misconception_id = read_tag(answer)
        cause = _CAUSE.get(outcome or "", "gap")
        follow_up = None if outcome == "correct" else f"Can you walk me through your reasoning on {question['id']}?"
        vote = {"cause": cause, "misconception_id": misconception_id, "follow_up": follow_up}
        return Diagnosis(cause=cause, misconception_id=misconception_id, follow_up=follow_up,
                         vote_share=1.0, votes=[vote, vote, vote])
