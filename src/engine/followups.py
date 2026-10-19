"""Turn flow (spec section 3): provisional grade, diagnosis, follow-ups, misconception tracking.

The final grade of an exchange is produced by the `update_scorecard` tool, which
regrades the whole exchange (main answer plus follow-ups).
"""

from __future__ import annotations

import re

GENERIC_FOLLOW_UP = "Could you explain your reasoning?"
FOLLOW_UP_CAUSES = ("misconception", "misread", "slip")   # a plain "gap" is never followed up
MISCONCEPTION_ID = re.compile(r"^(py|sql|ds)\.[a-z-]+\.[a-z0-9-]+$")


# --------------------------------------------------------------------------
# What the candidate is shown
# --------------------------------------------------------------------------

def render_table(table: dict) -> str:
    lines = [f"Table {table['name']}:", " | ".join(table["columns"])]
    lines += [" | ".join("NULL" if cell is None else str(cell) for cell in row) for row in table["rows"]]
    return "\n".join(lines)


def render_prompt(question: dict) -> str:
    """The question text word for word, then its code, then its tables as plain text."""
    parts = [question["question"]]
    if question.get("code"):
        parts.append(question["code"])
    parts += [render_table(t) for t in question.get("tables") or []]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Misconception tracking
# --------------------------------------------------------------------------

class MisconceptionTracker:
    """Counts the replies on which each misconception id won the diagnosis vote.

    An id that won once is suspected; at `confirmations` replies it is confirmed.
    Replies may come from different questions.
    """

    def __init__(self, confirmations: int = 2):
        self.confirmations = confirmations
        self.replies: dict[str, list[tuple[int, float]]] = {}   # id -> [(turn, vote_share), ...]

    def record(self, misconception_id: str, turn: int, vote_share: float = 1.0) -> None:
        self.replies.setdefault(misconception_id, []).append((turn, vote_share))

    def count(self, misconception_id: str | None) -> int:
        return len(self.replies.get(misconception_id, []))

    def is_confirmed(self, misconception_id: str | None) -> bool:
        return self.count(misconception_id) >= self.confirmations

    def ids_for_turn(self, turn: int) -> set[str]:
        """Misconception ids that won the vote on a reply of this main question."""
        return {m for m, replies in self.replies.items() if any(t == turn for t, _ in replies)}

    def confirmed(self) -> set[str]:
        return {m for m in self.replies if self.is_confirmed(m)}

    def suspected(self) -> set[str]:
        return {m for m in self.replies if not self.is_confirmed(m)}

    def findings(self) -> list[dict]:
        """One entry per id seen, in order of first sighting (schema: report.misconceptions)."""
        out = []
        for mid, replies in self.replies.items():
            out.append({"id": mid, "confirmed": self.is_confirmed(mid),
                        "evidence_turns": sorted({turn for turn, _ in replies}),
                        "vote_share": round(sum(share for _, share in replies) / len(replies), 3)})
        return out


# --------------------------------------------------------------------------
# One exchange
# --------------------------------------------------------------------------

def clean_diagnosis(diagnosis: dict, question: dict, known_ids: set[str]) -> dict:
    """Copy of a diagnosis where a misconception id the engine cannot trust is dropped.

    An id counts only if it has the right shape and is on the question's list or in the
    known misconception records. (Otherwise the result file would not match the schema.)
    """
    diagnosis = dict(diagnosis)
    mid = diagnosis.get("misconception_id")
    trusted = (isinstance(mid, str) and MISCONCEPTION_ID.match(mid)
               and (mid in question.get("misconceptions", []) or mid in known_ids))
    if diagnosis.get("cause") != "misconception" or not trusted:
        diagnosis["misconception_id"] = None
    diagnosis.setdefault("vote_share", 1.0)
    return diagnosis


class Exchange:
    """Everything between the main answer and the final grade, as steps.

    Creating it grades the main answer (the provisional grade) and, unless that is a 2 or
    diagnosis is off, diagnoses it. Then `pending` is the follow-up to ask next (or None
    when the exchange is complete); `answer_follow_up(text)` records its reply, diagnoses it
    where the system does, and decides what comes next. `grade(question, answer, follow_ups)`
    is the (cached) grader call.
    """

    def __init__(self, question: dict, prompt: str, answer: str, *, turn: int, diagnoser, config,
                 tracker: MisconceptionTracker, grade, known_ids: set[str]):
        self.question, self.turn, self.config = question, turn, config
        self.diagnoser, self.tracker, self.known_ids = diagnoser, tracker, known_ids
        self.main_exchange = {"prompt": prompt, "answer": answer}   # context for diagnosing a follow-up answer
        self.provisional = grade(question, answer, [])
        self.diagnosis: dict | None = None
        self.follow_ups: list[dict] = []
        self.pending: str | None = None      # follow-up waiting to be asked
        self._latest: dict | None = None     # newest diagnosis
        if self.provisional["score"] >= 2 or config.diagnosis == "off":
            return                           # a 2 is final; B1 never diagnoses
        self.diagnosis = self._latest = self._diagnose(answer, [])
        self._plan()

    def _diagnose(self, text: str, history: list) -> dict:
        diag = clean_diagnosis(self.diagnoser.diagnose(self.question, text, history), self.question,
                               self.known_ids)
        if diag.get("misconception_id"):
            self.tracker.record(diag["misconception_id"], self.turn, diag["vote_share"])
        return diag

    def _plan(self) -> None:
        """Decide the next follow-up from the newest diagnosis."""
        diag, config = self._latest, self.config
        self.pending = None
        if config.diagnosis == "single":
            # One diagnosis on the main reply; at most one generic follow-up, which is not diagnosed.
            if config.generic_follow_up and not self.follow_ups and diag["cause"] in FOLLOW_UP_CAUSES:
                self.pending = GENERIC_FOLLOW_UP
            return
        # structured: targeted follow-ups, diagnosing each answer
        if len(self.follow_ups) >= config.max_follow_ups:
            return
        if diag["cause"] not in FOLLOW_UP_CAUSES or not diag.get("follow_up"):
            return         # a plain gap (or nothing to ask)
        if self.tracker.is_confirmed(diag.get("misconception_id")):
            return         # misconception already confirmed
        self.pending = diag["follow_up"]

    @property
    def done(self) -> bool:
        return self.pending is None

    def answer_follow_up(self, text: str) -> None:
        """Record the reply to the pending follow-up and plan what comes next."""
        if self.pending is None:
            raise RuntimeError("no follow-up is waiting for an answer")
        follow_up = {"prompt": self.pending, "answer": text}
        self.follow_ups.append(follow_up)
        if self.config.diagnosis == "single":
            self.pending = None
            return
        diag = self._diagnose(text, [self.main_exchange] + self.follow_ups[:-1])
        follow_up["diagnosis"] = self._latest = diag
        self._plan()

    def result(self) -> dict:
        """{"provisional": Grade, "diagnosis": Diagnosis | None, "follow_ups": [FollowUp, ...]}"""
        return {"provisional": self.provisional, "diagnosis": self.diagnosis, "follow_ups": self.follow_ups}


def run_exchange(question: dict, prompt: str, answer: str, *, turn: int, respondent, diagnoser,
                 config, tracker: MisconceptionTracker, grade, known_ids: set[str]) -> dict:
    """Blocking version of `Exchange`: asks each follow-up of `respondent` until none is left."""
    exchange = Exchange(question, prompt, answer, turn=turn, diagnoser=diagnoser, config=config,
                        tracker=tracker, grade=grade, known_ids=known_ids)
    while exchange.pending is not None:
        exchange.answer_follow_up(
            respondent.reply(exchange.pending, question=question, kind="follow_up", turn=turn))
    return exchange.result()
