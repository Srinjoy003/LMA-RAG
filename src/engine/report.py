"""Fit decision and report (spec section 6)."""

from __future__ import annotations

import re

from .scorecard import MINIMUM_CUT

MEETS_AT = 0.8     # p >= 0.8: meets the minimum
BELOW_AT = 0.2     # p <= 0.2: below the minimum
MIN_QUESTIONS = 2  # fewer graded questions than this: always "uncertain"

LEVEL_NAMES = {0: "below level 1", 1: "level 1", 2: "level 2", 3: "level 3"}

# Wording that asks to change the rating or to see the rubric
_MANIPULATION = re.compile("|".join([
    r"\b(rate|score|grade) me\b",
    r"\b(change|raise|increase|upgrade|boost) (my|the) (rating|score|grade|level|result)\b",
    r"\bgive me (a |an )?(high|full|top|perfect|max|maximum|expert)",
    r"\bmark (this|it|me) as\b",
    r"\bignore (all |any |the |your |previous |prior |above )+(instructions|rules|prompt)",
    r"\b(rubric|answer key|model answer|marking scheme|scoring guide)\b",
    r"\b(i am|i'm|consider me) an? (expert|senior)\b",
]), re.IGNORECASE)


def find_manipulation(text: str) -> bool:
    return bool(_MANIPULATION.search(text or ""))


def skill_fit(skill_score, minimum: int) -> tuple[str, float]:
    """('meets' | 'below' | 'uncertain', p) for one skill against the job minimum."""
    p = skill_score.prob_at_least(MINIMUM_CUT[minimum])
    if skill_score.n < MIN_QUESTIONS:
        return "uncertain", p      # the prior alone must never decide
    if p >= MEETS_AT:
        return "meets", p
    if p <= BELOW_AT:
        return "below", p
    return "uncertain", p


def decide(fits: dict[str, str], minimums: dict[str, int], levels: dict[str, int]) -> tuple[str, str]:
    """(decision, plain-words reason) from each skill's fit."""
    below = [s for s, f in fits.items() if f == "below"]
    unsure = [s for s, f in fits.items() if f == "uncertain"]
    if below:
        parts = [f"{s} is below the job minimum (level {minimums[s]}; estimated {LEVEL_NAMES[levels[s]]})"
                 for s in below]
        return "not_fit", "; ".join(parts) + "."
    if unsure:
        parts = [f"cannot tell whether {s} reaches level {minimums[s]}" for s in unsure]
        return "needs_review", "Not enough evidence: " + "; ".join(parts) + "."
    return "fit", "Every skill meets the job minimum."


def build_report(state) -> dict:
    """The `report` part of the result, from the final interview state."""
    config, job = state.config, state.job
    minimums = {s: job.get("min_levels", {}).get(s, 0) for s in state.scores}
    levels, fits = {}, {}
    levels_report = {}
    for skill, score in state.scores.items():
        fit, _ = skill_fit(score, minimums[skill])
        fits[skill], levels[skill] = fit, score.level
        levels_report[skill] = {
            "level": score.level, "estimate": round(score.estimate, 4), "se": round(score.se, 4),
            "confident": score.se < config.uncertainty_threshold, "minimum": minimums[skill],
            "meets_minimum": {"meets": True, "below": False, "uncertain": None}[fit]}
    decision, reason = decide(fits, minimums, levels)

    required = list(state.required_topics)
    tested = [t for t in required if t in state.tested_topics]
    findings = state.tracker.findings()
    return {
        "levels": levels_report,
        "decision": decision,
        "decision_reason": reason,
        "misconceptions": findings,
        "probed_misconceptions": sorted(state.probed_misconceptions()),
        "topic_coverage": {"required": required, "tested": tested,
                           "untested": [t for t in required if t not in state.tested_topics]},
        "confidence_style": None,
        "flags": list(state.flags),
        "summary": summarise(state, levels_report, decision, reason, findings),
    }


def summarise(state, levels_report: dict, decision: str, reason: str, findings: list[dict]) -> str:
    skills = ", ".join(f"{s} {LEVEL_NAMES[r['level']]} (needs {r['minimum']})" for s, r in levels_report.items())
    text = f"Recommendation: {decision.replace('_', ' ')}. {reason} Estimated levels: {skills}."
    records = state.misconception_records
    for finding in findings:
        what = records.get(finding["id"], {}).get("description", finding["id"])
        kind = "Confirmed" if finding["confirmed"] else "Suspected (seen once, not confirmed)"
        text += f" {kind} misconception: {what}."
    if state.flags:
        text += " Flags: " + "; ".join(state.flags) + "."
    return text
