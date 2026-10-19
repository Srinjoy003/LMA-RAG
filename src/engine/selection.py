"""Choosing the next question and deciding when to stop (spec sections 4 and 5).

`InterviewState` holds everything the rules need. The ranking and the seeded
tie-break live here, not in the question index: the index only filters.
"""

from __future__ import annotations

import random

from shared.interfaces import SKILLS

from . import report
from .followups import MisconceptionTracker
from .scorecard import DIFFICULTY, SkillScore, nearest_level

ALL_QUESTIONS = 100_000    # "k" for a search that returns every eligible question


class InterviewState:
    def __init__(self, job: dict, config, components, required_topics: list[str], rng: random.Random):
        self.job = job
        self.config = config
        self.source = components.question_source
        self.generator = components.question_generator
        self.misconception_records = components.misconceptions
        self.required_topics = list(required_topics)
        self.rng = rng

        self.scores = {skill: SkillScore(config.start_level) for skill in SKILLS}
        self.tracker = MisconceptionTracker(config.misconception_confirmations)
        self.asked: list[dict] = []          # questions chosen so far (the last may still be open)
        self.asked_ids: set[str] = set()
        self.tested_topics: set[str] = set()
        self.exhausted: set[str] = set()
        self.pending: dict | None = None     # chosen question waiting for update_scorecard
        self.flags: list[str] = []
        self.errors: list[str] = []
        self.grade_cache: dict = {}
        self.graded: list[dict] = []         # {"turn", "skill", "topics", "score"} per graded main question

    def probed_misconceptions(self) -> set[str]:
        """Misconceptions listed on any main question asked so far (the open one included)."""
        return {m for q in self.asked for m in q.get("misconceptions", [])}

    def weak_topics(self, skill: str) -> set[str]:
        """Topics of this skill where a main question scored below 2 and no misconception was
        confirmed on it: a misconception may still hide there (the probe rule)."""
        weak: set[str] = set()
        for item in self.graded:
            if item["skill"] != skill or item["score"] >= 2:
                continue
            if any(self.tracker.is_confirmed(m) for m in self.tracker.ids_for_turn(item["turn"])):
                continue
            weak.update(item["topics"])
        return weak

    @property
    def n_graded(self) -> int:
        return sum(score.n for score in self.scores.values())

    def count_asked(self, skill: str) -> int:
        return sum(1 for q in self.asked if q["skill"] == skill)

    def untested_required(self, skill: str) -> list[str]:
        return [t for t in self.required_topics
                if t.startswith(skill + "-") and t not in self.tested_topics]


# --------------------------------------------------------------------------
# Skill and level
# --------------------------------------------------------------------------

def next_target(state: InterviewState) -> tuple[str, int, str] | None:
    """(skill, level, why) for the next main question, or None if nothing is left to ask."""
    config = state.config
    if config.mode == "fixed":
        open_skills = [s for s in config.skill_order
                       if s not in state.exhausted and state.count_asked(s) < len(config.fixed_levels)]
        if not open_skills:
            return None
        # rotation: the skill with the fewest questions, ties in skill_order
        skill = min(open_skills, key=lambda s: (state.count_asked(s), config.skill_order.index(s)))
        n = state.count_asked(skill)
        return skill, config.fixed_levels[n], (
            f"fixed order: question {n + 1} of {skill} is level {config.fixed_levels[n]}")

    open_skills = [s for s in SKILLS if s not in state.exhausted]
    if not open_skills:
        return None
    # highest SE first; ties in the order py, sql, ds
    skill = max(open_skills, key=lambda s: (round(state.scores[s].se, 9), -SKILLS.index(s)))
    score = state.scores[skill]
    level = nearest_level(score.estimate)
    return skill, level, (f"{skill} has the highest uncertainty (SE {score.se:.2f}); "
                          f"level {level} is nearest its estimate {score.estimate:+.2f}")


# --------------------------------------------------------------------------
# Question
# --------------------------------------------------------------------------

def _rank_probing(state: InterviewState, skill: str, pool: list[dict]) -> tuple[dict, str]:
    """Ranking with `probe_missed_topics` on (spec section 4), best first:
    (a) lists a suspected misconception; (b) probe: shares a topic with an earlier weak question
    of this skill and lists a misconception not yet probed; (c) job-aware: covers an untested
    required topic; (d) lists more not-yet-probed misconceptions. Then the seeded tie-break."""
    untested = set(state.untested_required(skill)) if state.config.retrieval == "job-aware" else set()
    suspected = state.tracker.suspected()
    weak = state.weak_topics(skill)
    probed = state.probed_misconceptions()

    def facts(q: dict) -> tuple[bool, bool, bool, int]:
        fresh = set(q.get("misconceptions", [])) - probed
        return (bool(suspected & set(q.get("misconceptions", []))),
                bool(weak & set(q.get("topics", []))) and bool(fresh),
                bool(untested & set(q.get("topics", []))),
                len(fresh))

    def key(q: dict) -> tuple[bool, bool, bool, int]:
        a, b, c, d = facts(q)
        return (not a, not b, not c, -d)

    best = min(key(q) for q in pool)
    top = sorted((q for q in pool if key(q) == best), key=lambda q: q["id"])
    question = state.rng.choice(top)
    a, b, c, d = facts(question)
    fresh = sorted(set(question.get("misconceptions", [])) - probed)
    if a:
        why = "lists a suspected misconception to confirm: " + ", ".join(
            sorted(suspected & set(question["misconceptions"])))
    elif b:
        why = ("probes a topic where an earlier answer was weak, for a misconception not yet probed: "
               + ", ".join(sorted(weak & set(question["topics"]))) + " (" + ", ".join(fresh) + ")")
    elif c:
        why = "covers required topics not yet tested: " + ", ".join(sorted(untested & set(question["topics"])))
    elif d:
        why = "lists misconceptions not yet probed: " + ", ".join(fresh)
    else:
        why = "no preferred question available; picked among the rest"
    return question, why


def _rank(state: InterviewState, skill: str, pool: list[dict]) -> tuple[dict, str]:
    """Best question of the pool: job-aware first, then suspected misconception, then any
    (with `probe_missed_topics`, see `_rank_probing`).
    Ties are broken with the interview's random generator (candidates sorted by id first)."""
    if state.config.probe_missed_topics:
        return _rank_probing(state, skill, pool)
    untested = set(state.untested_required(skill)) if state.config.retrieval == "job-aware" else set()
    suspected = state.tracker.suspected()

    def tier(q: dict) -> int:
        if untested & set(q.get("topics", [])):
            return 0
        if suspected & set(q.get("misconceptions", [])):
            return 1
        return 2

    best = min(tier(q) for q in pool)
    top = sorted((q for q in pool if tier(q) == best), key=lambda q: q["id"])
    question = state.rng.choice(top)
    if best == 0:
        topics = sorted(untested & set(question["topics"]))
        why = "covers required topics not yet tested: " + ", ".join(topics)
    elif best == 1:
        mids = sorted(suspected & set(question["misconceptions"]))
        why = "lists a suspected misconception to confirm: " + ", ".join(mids)
    else:
        why = "no preferred question available; picked among the rest"
    return question, why


def _eligible(state: InterviewState, skill: str, level: int) -> list[dict]:
    """Unasked bank questions of this skill and level (every one, in the source's order)."""
    prefer = state.untested_required(skill) if state.config.retrieval == "job-aware" else None
    found = state.source.search(skill=skill, levels=[level], prefer_topics=prefer,
                                exclude_ids=frozenset(state.asked_ids), k=ALL_QUESTIONS)
    return [q for q, _ in found
            if q["skill"] == skill and q["level"] == level and q["id"] not in state.asked_ids]


def _can_generate(state: InterviewState, skill: str) -> bool:
    return bool(state.config.allow_generated_questions and state.generator
                and state.untested_required(skill))


def _generate(state: InterviewState, skill: str, level: int) -> dict | None:
    """Fallback 3: a generated question on an untested required topic of this skill."""
    topics = state.untested_required(skill)
    if not _can_generate(state, skill):
        return None
    try:
        question = dict(state.generator.generate(skill, level, topics[0]))
    except Exception as e:   # a failing generator must not stop the interview
        state.errors.append(f"question generator failed: {type(e).__name__}: {e}")
        return None
    question.setdefault("id", f"GEN-{skill}-L{level}-{len(state.asked) + 1:03d}")
    question.setdefault("skill", skill)
    question.setdefault("level", level)
    question.setdefault("topics", [topics[0]])
    question.setdefault("source", "generated-live")
    return question


def pick_question(state: InterviewState, skill: str, level: int, probe: bool = False) -> dict | None:
    """Question for this skill and level, with the fallbacks of section 4.
    Marks the skill exhausted and returns None when nothing is left.
    With `probe`, a question would only be generated, not generated for real."""
    estimate = state.scores[skill].estimate
    others = sorted((lv for lv in DIFFICULTY if lv != level),
                    key=lambda lv: (abs(DIFFICULTY[lv] - estimate), lv))   # nearer the estimate first
    for lv in [level] + others:
        pool = _eligible(state, skill, lv)
        if pool:
            question, why = _rank(state, skill, pool)
            note = "" if lv == level else f"no unasked level-{level} questions left, used level {lv}; "
            return {"question": question, "source": "bank", "reason": note + why}
    if probe and _can_generate(state, skill):
        return {"probe": True}
    question = _generate(state, skill, level)
    if question:
        return {"question": question, "source": "generated",
                "reason": f"bank has nothing left for {skill}; generated a question on {question['topics'][0]}"}
    state.exhausted.add(skill)
    return None


def choose_next(state: InterviewState, probe: bool = False) -> dict | None:
    """The next main question as {"skill", "level", "target_level", "question", "source", "reason"},
    or None when nothing is left. (It does not record the question as asked.)"""
    while True:
        target = next_target(state)
        if target is None:
            return None
        skill, level, target_reason = target
        picked = pick_question(state, skill, level, probe)
        if picked is None:
            continue    # that skill ran dry; try the next one
        if probe:
            return picked
        question = picked["question"]
        return {"skill": skill, "level": question["level"], "target_level": level,
                "question": question, "source": picked["source"],
                "reason": f"{target_reason}; {picked['reason']}"}


def has_next(state: InterviewState) -> bool:
    """Is there a question left to ask? Changes nothing except noting exhausted skills."""
    rng_state = state.rng.getstate()
    choice = choose_next(state, probe=True)
    state.rng.setstate(rng_state)
    return choice is not None


def record_asked(state: InterviewState, choice: dict) -> None:
    question = choice["question"]
    state.asked.append(question)
    state.asked_ids.add(question["id"])
    state.tested_topics.update(question.get("topics", []))
    state.pending = question


# --------------------------------------------------------------------------
# Stopping
# --------------------------------------------------------------------------

def _settled(state: InterviewState, skill: str) -> bool:
    """Is this skill done as far as the stop rule is concerned?"""
    if skill in state.exhausted:
        return True
    score, config = state.scores[skill], state.config
    if score.n < 2:
        return False
    if config.stop_rule == "decision":
        minimum = state.job.get("min_levels", {}).get(skill, 0)
        return report.skill_fit(score, minimum)[0] != "uncertain"
    return score.se < config.uncertainty_threshold


def check_stop(state: InterviewState) -> str | None:
    """The stop reason if the interview may end now, else None (section 5)."""
    config, n = state.config, state.n_graded
    if config.stop_rule == "fixed_count" and n >= config.fixed_count:
        return "fixed_length"
    if config.stop_rule in ("uncertainty", "decision") and all(_settled(state, s) for s in SKILLS):
        unmeasured = [s for s in state.exhausted
                      if not (state.scores[s].n >= 2 and state.scores[s].se < config.uncertainty_threshold)]
        return "bank_exhausted" if unmeasured else "confident"
    if config.stop_rule != "fixed_count" and n >= config.max_questions:
        return "max_questions"
    if has_next(state):
        return None
    # nothing left to ask: every skill ran dry, or the fixed order is complete
    return "bank_exhausted" if len(state.exhausted) == len(SKILLS) else "fixed_length"
