"""Step-based interview session (spec section 1): one interview as a state machine.

A blocking driver (`run_interview`) cannot serve a web app, which only gets the answer
when the person presses Submit. The session instead hands out one prompt at a time:

    session = InterviewSession(job, config, components)
    while (prompt := session.next_prompt()) is not None:
        session.submit(ask_the_person(prompt["text"]))
    result = session.result()

The prompts are the main questions and the follow-ups, in the order the rules choose them.
The session never reads anything but the answer text.
"""

from __future__ import annotations

import copy
import random
from typing import Any, TypedDict

from . import report
from .followups import Exchange
from .result import build_result, build_turn, now, required_topics_for
from .selection import InterviewState, check_stop
from .tools import InterviewTools


class Prompt(TypedDict):
    kind: str          # "main" or "follow_up"
    turn: int          # number of the main question (a follow-up has its main question's number)
    text: str          # exactly what is shown to the candidate
    question: dict     # the question being asked, for display (code, tables)
    skill: str
    level: int


class InterviewSession:
    def __init__(self, job: dict, config, components, *, candidate_id: str | None = None,
                 rng: random.Random | None = None):
        if config.driver != "code":
            raise ValueError("InterviewSession is the code driver; the agent driver is in engine.agent")
        if config.diagnosis != "off" and components.diagnoser is None:
            raise ValueError(f"diagnosis={config.diagnosis!r} needs a diagnoser in Components")

        self.job, self.config, self.components, self.candidate_id = job, config, components, candidate_id
        self.rng = rng or random.Random(config.seed)
        self.started = now()
        self.errors: list[str] = []
        self.required = required_topics_for(job, config, components, self.errors)
        self.state = InterviewState(job, config, components, self.required, self.rng)
        self.state.errors = self.errors
        self.tools = InterviewTools(self.state, components.grader)
        self._known_ids = set(components.misconceptions)

        self.turns: list[dict] = []              # finished main questions
        self.stop_reason: str | None = None
        self._report: dict | None = None
        self._finished_at = None
        self._finished = False
        self._prompt: Prompt | None = None       # waiting for an answer
        self._found: dict | None = None          # the open main question (search_question_bank result)
        self._answer: str | None = None          # its main answer
        self._exchange: Exchange | None = None   # its follow-up flow, once the main answer is in
        self._advance()

    # ---- the public API --------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self._finished

    def next_prompt(self) -> Prompt | None:
        """The prompt waiting for an answer (asking for it again gives the same one), or None
        when the interview is over."""
        return self._prompt

    def submit(self, answer: str) -> None:
        """Answer the current prompt. Afterwards `next_prompt()` gives the follow-up or the next
        question, or None if the interview ended."""
        if self._prompt is None:
            raise RuntimeError("the interview is over" if self._finished else "no question is waiting for an answer")
        answer = answer or ""
        try:
            if self._prompt["kind"] == "main":
                self._answer = answer
                self._exchange = Exchange(self._found_question(), self._found["prompt"], answer,
                                          turn=self._found["turn"], diagnoser=self.components.diagnoser,
                                          config=self.config, tracker=self.state.tracker,
                                          grade=self.tools.grade, known_ids=self._known_ids)
            else:
                self._exchange.answer_follow_up(answer)
            self._prompt = None
            if self._exchange.pending is not None:
                self._prompt = self._make_prompt("follow_up", self._exchange.pending)
            else:
                self._close_turn()
                self._advance()
        except Exception as e:
            self._fail(e)

    def result(self) -> dict:
        """The interview result (schema: interview_result.schema.json). Only for a finished interview."""
        if not self._finished:
            raise RuntimeError("the interview is not finished")
        return copy.deepcopy(build_result(
            job=self.job, config=self.config, candidate_id=self.candidate_id, started=self.started,
            finished=self._finished_at, required=self.required, turns=self.turns,
            stop_reason=self.stop_reason, report=self._report, errors=self.errors, tool_log=self.tools.log))

    def scorecard_view(self) -> dict[str, dict[str, Any]]:
        """Per skill: estimate, SE, level and number of graded questions so far, with the job
        minimum and whether the SE is below the stopping threshold (for the interviewer's screen)."""
        minimums = self.job.get("min_levels", {})
        view = {}
        for skill, score in self.state.scores.items():
            view[skill] = {**score.state(), "minimum": minimums.get(skill, 0),
                           "confident": score.se < self.config.uncertainty_threshold}
        return view

    def interviewer_view(self) -> dict[str, Any]:
        """Everything the interviewer's screen shows besides the scorecard: the diagnoses and
        misconceptions found so far, and the open question's provisional grade."""
        exchange = self._exchange
        return {
            "scorecard": self.scorecard_view(),
            "misconceptions": self.state.tracker.findings(),
            "flags": list(self.state.flags),
            "current": None if exchange is None else {
                "provisional_grade": exchange.provisional, "diagnosis": exchange.diagnosis,
                "follow_ups": copy.deepcopy(exchange.follow_ups)},
        }

    @property
    def current_exchange(self) -> dict | None:
        """The open main question's main answer and follow-ups so far (None between questions)."""
        if self._found is None or self._answer is None:
            return None
        return {"turn": self._found["turn"], "question_id": self._found["question_id"],
                "answer": self._answer,
                "follow_ups": copy.deepcopy(self._exchange.follow_ups) if self._exchange else []}

    # ---- internals -------------------------------------------------------------------
    def _found_question(self) -> dict:
        return self.state.pending

    def _make_prompt(self, kind: str, text: str) -> Prompt:
        question = self._found_question()
        return Prompt(kind=kind, turn=self._found["turn"], text=text, question=copy.deepcopy(question),
                      skill=self._found["skill"], level=self._found["level"])

    def _advance(self) -> None:
        """Ask for the next main question, or end the interview if the stop rule is met."""
        try:
            self._found = self._answer = self._exchange = None
            if check_stop(self.state) is None:
                found = self.tools.search_question_bank()
                if not found["ok"]:
                    raise RuntimeError(f"search_question_bank refused: {found['error']}")
                self._found = found
                self._prompt = self._make_prompt("main", found["prompt"])
                return
            ended = self.tools.end_interview()
            if not ended["ok"]:
                raise RuntimeError(f"end_interview refused: {ended['error']}")
            self._prompt = None
            self.stop_reason, self._report = self.tools.stop_reason, self.tools.report
            self._finish()
        except Exception as e:
            self._fail(e)

    def _close_turn(self) -> None:
        """Final grade and scorecard update for the open question; log the turn."""
        exchange = self._exchange.result()
        update = self.tools.update_scorecard(self._found["question_id"], self._answer, exchange["follow_ups"])
        if not update["ok"]:
            raise RuntimeError(f"update_scorecard refused: {update['error']}")
        self.turns.append(build_turn(self._found, self._answer, exchange, update))

    def fail(self, error: Exception) -> None:
        """End the interview because something outside the session failed (for example the
        respondent). Same result as an internal error: stop_reason "error" and a report from
        what was scored so far."""
        if not self._finished:
            self._fail(error)

    def _fail(self, error: Exception) -> None:
        """An exception ends the interview with a report from what was scored so far."""
        self.errors.append(f"{type(error).__name__}: {error}")
        self._prompt = None
        self.stop_reason, self._report = "error", report.build_report(self.state)
        self._finish()

    def _finish(self) -> None:
        self._finished = True
        self._finished_at = now()
