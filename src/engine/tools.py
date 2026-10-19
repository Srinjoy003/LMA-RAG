"""The three tools of the interview (spec section 1) and the rules they enforce.

The code driver calls them directly; an LLM driver would call the same ones.
Every call is logged. A call that breaks a rule is refused: it returns
{"ok": False, "error": ...} and changes nothing.
"""

from __future__ import annotations

from shared.interfaces import SKILLS

from . import report
from .followups import render_prompt
from .selection import InterviewState, check_stop, choose_next, record_asked


class ToolError(Exception):
    """A call that the rules refuse."""


class InterviewTools:
    def __init__(self, state: InterviewState, grader):
        self.state = state
        self.grader = grader
        self.log: list[dict] = []     # {"turn", "name", "args", "ok", "error"}
        self.stop_reason: str | None = None
        self.report: dict | None = None
        self.choice: dict | None = None   # the choice behind the open question

    # ---- shared helpers ------------------------------------------------------
    def _run(self, name: str, args: dict, turn: int, action) -> dict:
        try:
            result = action()
        except ToolError as e:
            self.log.append({"turn": turn, "name": name, "args": args, "ok": False, "error": str(e)})
            return {"ok": False, "error": str(e)}
        except Exception as e:
            self.log.append({"turn": turn, "name": name, "args": args, "ok": False,
                             "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"turn": turn, "name": name, "args": args, "ok": True, "error": None})
        return {"ok": True, **result}

    def grade(self, question: dict, answer: str, follow_ups: list) -> dict:
        """Grade an exchange. The same exchange is graded once (a main answer with no
        follow-ups gets the same grade as its provisional one)."""
        key = (question["id"], answer, tuple(f["answer"] for f in follow_ups))
        cache = self.state.grade_cache
        if key not in cache:
            grade = dict(self.grader.grade(question, answer, list(follow_ups)))
            if grade.get("score") not in (0, 1, 2) or not grade.get("votes"):
                raise ValueError(f"grader returned an invalid grade: {grade}")
            grade.setdefault("agreement", grade["votes"].count(grade["score"]) / len(grade["votes"]))
            cache[key] = grade
        return dict(cache[key])

    # ---- search_question_bank --------------------------------------------------
    def search_question_bank(self, skill: str | None = None, level: int | None = None) -> dict:
        """Return the next question. `skill` and `level` are optional; if given they must
        match what the rules choose, otherwise the call is refused."""
        state = self.state
        args = {k: v for k, v in (("skill", skill), ("level", level)) if v is not None}
        turn = len(state.asked) + (0 if state.pending else 1)

        def action() -> dict:
            if state.pending:
                raise ToolError(f"question {state.pending['id']} is still open; call update_scorecard first")
            if check_stop(state):
                raise ToolError("the stop rule is met; call end_interview")
            if skill is not None and skill not in SKILLS:
                raise ToolError(f"unknown skill {skill!r}")
            if level is not None and level not in (1, 2, 3):
                raise ToolError(f"unknown level {level!r}")
            rng_state = state.rng.getstate()
            choice = choose_next(state)
            if choice is None:
                raise ToolError("no questions are left to ask; call end_interview")
            if skill is not None and skill != choice["skill"]:
                state.rng.setstate(rng_state)
                raise ToolError(f"the rules choose skill {choice['skill']}, not {skill}")
            if level is not None and level != choice["target_level"]:
                state.rng.setstate(rng_state)
                raise ToolError(f"the rules choose level {choice['target_level']} for {choice['skill']}, "
                                f"not {level}")
            record_asked(state, choice)
            self.choice = choice
            question = choice["question"]
            return {"turn": len(state.asked), "question_id": question["id"], "skill": choice["skill"],
                    "level": choice["level"], "source": choice["source"],
                    "topics": list(question.get("topics", [])),
                    "selection_reason": choice["reason"], "prompt": render_prompt(question)}

        return self._run("search_question_bank", args, turn, action)

    # ---- update_scorecard ---------------------------------------------------------
    def update_scorecard(self, question_id: str, answer: str, follow_ups: list | None = None) -> dict:
        """Grade the whole exchange for the question that is open, and update its skill's scorecard."""
        state, follow_ups = self.state, list(follow_ups or [])
        args = {"question_id": question_id, "n_follow_ups": len(follow_ups)}

        def action() -> dict:
            if state.pending is None:
                raise ToolError("no question is open")
            if question_id != state.pending["id"]:
                raise ToolError(f"the open question is {state.pending['id']}, not {question_id}")
            question = state.pending
            grade = self.grade(question, answer, follow_ups)
            weight = grade["agreement"]
            if self.choice["source"] == "generated":
                weight *= state.config.generated_weight
                flag = f"generated question used (question {len(state.asked)}: {question['id']})"
                if flag not in state.flags:
                    state.flags.append(flag)
            state.scores[question["skill"]].update(question["level"], grade["score"], weight)
            state.graded.append({"turn": len(state.asked), "skill": question["skill"],
                                 "topics": list(question.get("topics", [])), "score": grade["score"]})
            if any(report.find_manipulation(text) for text in [answer] + [f["answer"] for f in follow_ups]):
                state.flags.append(f"possible manipulation attempt in question {len(state.asked)}: "
                                   "the answer asks to change the rating or to see the rubric")
            state.pending = None
            return {"grade": grade, "evidence_weight": round(weight, 4),
                    "scorecard": {s: sc.state() for s, sc in state.scores.items()}}

        return self._run("update_scorecard", args, len(state.asked), action)

    # ---- end_interview -------------------------------------------------------------
    def end_interview(self) -> dict:
        """Produce the report. Refused until the stop rule is met."""
        state = self.state

        def action() -> dict:
            if state.pending:
                raise ToolError(f"question {state.pending['id']} is still open; call update_scorecard first")
            reason = check_stop(state)
            if reason is None:
                raise ToolError(f"the stop rule ({state.config.stop_rule}) is not met yet: "
                                f"{state.n_graded} main questions asked")
            self.stop_reason = reason
            self.report = report.build_report(state)
            return {"stop_reason": reason, "report": self.report}

        return self._run("end_interview", {}, len(state.asked), action)
