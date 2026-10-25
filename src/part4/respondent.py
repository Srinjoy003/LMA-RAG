"""A simulated candidate in an interview (interfaces.Respondent): key + answer rules + simulator.

For every reply, code plans the outcome from the candidate's key (the reference rules in
shared.stubs.reference_plan_answer) and the simulator words it. A planned safety attack is
appended word for word. `truths` logs every plan for the experiment runner; the engine never
reads it.
"""

from __future__ import annotations

import random

from shared.stubs import reference_plan_answer


class SimulatedRespondent:
    def __init__(self, candidate: dict, simulator, misconceptions: dict):
        self.candidate, self.simulator, self.misconceptions = candidate, simulator, misconceptions
        self.rng = random.Random(candidate.get("seed", 0))
        self.truths: list[tuple[int, str, dict]] = []
        self._main_plan: dict | None = None
        self._history: list[dict] = []        # this question's exchanges so far, for follow-ups

    def reply(self, prompt: str, *, question: dict | None, kind: str, turn: int) -> str:
        if question is None:
            return "Sorry, I'm not sure what you mean."
        if kind == "main":
            self._history = []
        plan = reference_plan_answer(self.candidate, question, kind=kind, turn=turn,
                                     previous=self._main_plan if kind == "follow_up" else None, rng=self.rng)
        if kind == "main":
            self._main_plan = plan
        self.truths.append((turn, kind, plan))
        text = self.simulator.answer(self.candidate, question, prompt, plan, history=list(self._history),
                                     misconceptions=self.misconceptions)
        if plan.get("attack"):
            text = f"{text} {plan['attack']}"
        self._history.append({"prompt": prompt, "answer": text})
        return text
