"""Checks for the question bank (spec sections 2, 4, 6, 9, 10, 13).

`validate_bank(bank)` returns a Report with errors and warnings. The thresholds
for the coverage rules are in `Rules`; the defaults are the spec's numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator

from .bank import LEVELS, SKILLS, Bank
from .execute import TIMEOUT_SECONDS, check_question


@dataclass
class Rules:
    """Coverage rules from spec section 6 (per skill). Every level of a skill holds the same
    number of questions, at least `min_questions_per_level`, with `explain_share` of them
    explain questions (6 of 10, 9 of 15) and the rest read-code."""
    min_questions_per_level: int = 10
    explain_share: float = 0.6
    min_topics_per_level: int = 4
    min_questions_per_topic: int = 2
    min_questions_per_required_topic: int = 3
    min_questions_per_misconception: int = 2


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def _label(item, index: int, kind: str) -> str:
    if isinstance(item, dict) and isinstance(item.get("id"), str):
        return item["id"]
    return f"{kind} #{index}"


def _schema_validator(root: Path, name: str) -> Draft202012Validator:
    with open(root / "schemas" / name, encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


def _format_path(error) -> str:
    return ".".join(str(p) for p in error.absolute_path) or "(item)"


# --------------------------------------------------------------------------
# a. schema
# --------------------------------------------------------------------------
def check_schema(bank: Bank, report: Report, questions: list, misconceptions: list):
    qv = _schema_validator(bank.root, "question.schema.json")
    mv = _schema_validator(bank.root, "misconception.schema.json")
    for i, q in enumerate(questions):
        for err in sorted(qv.iter_errors(q), key=lambda e: list(e.absolute_path)):
            report.error(f"[schema] {_label(q, i, 'question')}: {_format_path(err)}: {err.message[:200]}")
    for i, m in enumerate(misconceptions):
        for err in sorted(mv.iter_errors(m), key=lambda e: list(e.absolute_path)):
            report.error(f"[schema] {_label(m, i, 'misconception')}: {_format_path(err)}: {err.message[:200]}")


# --------------------------------------------------------------------------
# b. ids
# --------------------------------------------------------------------------
def check_ids(report: Report, questions: list, misconceptions: list):
    for kind, items in (("question", questions), ("misconception", misconceptions)):
        counts = Counter(x.get("id") for x in items)
        for id_, n in counts.items():
            if n > 1:
                report.error(f"[ids] duplicate {kind} id {id_!r} ({n} times)")

    for q in questions:
        id_, skill, level = q.get("id"), q.get("skill"), q.get("level")
        parts = id_.split("-") if isinstance(id_, str) else []
        if len(parts) != 3:
            continue  # the schema check already reports a malformed id
        if parts[0] != str(skill).upper():
            report.error(f"[ids] {id_}: id prefix {parts[0]!r} does not match skill {skill!r}")
        if parts[1] != f"L{level}":
            report.error(f"[ids] {id_}: id level {parts[1]!r} does not match level {level!r}")

    for m in misconceptions:
        id_ = m.get("id")
        parts = id_.split(".") if isinstance(id_, str) else []
        if len(parts) != 3:
            continue
        if parts[0] != m.get("skill"):
            report.error(f"[ids] {id_}: id starts with {parts[0]!r} but skill is {m.get('skill')!r}")
        if f"{m.get('skill')}-{parts[1]}" != m.get("topic"):
            report.error(f"[ids] {id_}: id topic part {parts[1]!r} does not match topic {m.get('topic')!r}")


# --------------------------------------------------------------------------
# c. references
# --------------------------------------------------------------------------
def check_references(bank: Bank, report: Report, questions: list, misconceptions: list):
    all_misc = bank.misconception_by_id
    for q in questions:
        id_, skill = q.get("id"), q.get("skill")
        for t in q.get("topics", []) or []:
            owner = bank.topic_skill(t)
            if owner is None:
                report.error(f"[refs] {id_}: topic {t!r} is not in topics.json")
            elif owner != skill:
                report.error(f"[refs] {id_}: topic {t!r} belongs to skill {owner!r}, not {skill!r}")
        for mid in q.get("misconceptions", []) or []:
            m = all_misc.get(mid)
            if m is None:
                report.error(f"[refs] {id_}: misconception {mid!r} does not exist")
            elif m.get("skill") != skill:
                report.error(f"[refs] {id_}: misconception {mid!r} has skill {m.get('skill')!r}, not {skill!r}")

    for m in misconceptions:
        t = m.get("topic")
        owner = bank.topic_skill(t) if isinstance(t, str) else None
        if owner is None:
            report.error(f"[refs] {m.get('id')}: topic {t!r} is not in topics.json")
        elif owner != m.get("skill"):
            report.error(f"[refs] {m.get('id')}: topic {t!r} belongs to skill {owner!r}, not {m.get('skill')!r}")

    known = bank.topic_ids()
    for job in bank.jobs:
        for t in job.get("required_topics", []):
            if t not in known:
                report.error(f"[refs] job {job.get('id')}: required topic {t!r} is not in topics.json")


# --------------------------------------------------------------------------
# d. coverage (per skill)
# --------------------------------------------------------------------------
def check_coverage(bank: Bank, report: Report, skills: list, rules: Rules):
    required = bank.required_topics()
    for skill in skills:
        qs = [q for q in bank.questions if isinstance(q, dict) and q.get("skill") == skill]
        if not qs:
            report.warn(f"[coverage] skill {skill!r}: no questions yet, coverage checks skipped")
            continue

        sizes = [sum(q.get("level") == level for q in qs) for level in LEVELS]
        if len(set(sizes)) > 1:
            report.error(f"[coverage] {skill}: levels 1-3 have {', '.join(map(str, sizes))} questions; "
                         "every level needs the same number")

        for level in LEVELS:
            cell = [q for q in qs if q.get("level") == level]
            where = f"{skill} level {level}"
            if len(cell) < rules.min_questions_per_level:
                report.error(f"[coverage] {where}: {len(cell)} questions, need at least "
                             f"{rules.min_questions_per_level}")
            n_explain = sum(q.get("type") == "explain" for q in cell)
            n_code = sum(q.get("type") == "read-code" for q in cell)
            want_explain = round(rules.explain_share * len(cell))
            if (n_explain, n_code) != (want_explain, len(cell) - want_explain):
                report.error(f"[coverage] {where}: {n_explain} explain and {n_code} read-code questions, "
                             f"expected {want_explain} and {len(cell) - want_explain}")
            distinct = {t for q in cell for t in q.get("topics", [])}
            if len(distinct) < rules.min_topics_per_level:
                report.error(f"[coverage] {where}: {len(distinct)} distinct topics, need at least "
                             f"{rules.min_topics_per_level}")

        topic_counts = Counter(t for q in qs for t in q.get("topics", []))
        for topic in sorted(bank.topic_ids(skill)):
            n = topic_counts[topic]
            if n < rules.min_questions_per_topic:
                report.error(f"[coverage] topic {topic}: {n} questions, need at least "
                             f"{rules.min_questions_per_topic}")
            if topic in required and n < rules.min_questions_per_required_topic:
                report.error(f"[coverage] job-required topic {topic}: {n} questions, need at least "
                             f"{rules.min_questions_per_required_topic}")

        misc_counts = Counter(m for q in qs for m in q.get("misconceptions", []))
        for m in bank.misconceptions_for(skill):
            n = misc_counts[m.get("id")]
            if n < rules.min_questions_per_misconception:
                report.error(f"[coverage] misconception {m.get('id')}: referenced by {n} questions, "
                             f"need at least {rules.min_questions_per_misconception}")


# --------------------------------------------------------------------------
# e. warnings
# --------------------------------------------------------------------------
def check_warnings(bank: Bank, report: Report, questions: list):
    for job in bank.jobs:
        if not str(job.get("description", "")).strip():
            report.warn(f"[jobs] job {job.get('id')}: description is empty")
    for q in questions:
        if q.get("type") == "read-code" and "expected_output" not in q:
            report.warn(f"[exec] {q.get('id')}: read-code question has no expected_output")


# --------------------------------------------------------------------------
# f. execution
# --------------------------------------------------------------------------
def check_execution(report: Report, questions: list, timeout: float = TIMEOUT_SECONDS, workers: int = 4):
    runnable = [q for q in questions
                if q.get("type") == "read-code" and "expected_output" in q and q.get("code")
                and q.get("skill") in SKILLS]
    if not runnable:
        return

    def run(q):
        try:
            return q, check_question(q, timeout)
        except Exception as e:  # a malformed question must not stop the whole run
            return q, f"could not run: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, runnable))
    for q, problem in results:
        if problem:
            report.error(f"[exec] {q['id']}: {problem}")
    report.info.append(f"executed {len(runnable)} read-code questions")


# --------------------------------------------------------------------------
# g. near duplicates
# --------------------------------------------------------------------------
def check_dupes(bank: Bank, report: Report, questions: list, embed_fn=None, threshold: float = 0.9):
    from .bank import question_text
    from .index import default_embedder, find_near_duplicates

    if len(questions) < 2:
        return
    embed_fn = embed_fn or default_embedder()
    ordered = sorted(questions, key=lambda q: q["id"])
    embeddings = embed_fn([question_text(q, bank.topic_name) for q in ordered])
    for a, b, sim in find_near_duplicates([q["id"] for q in ordered], embeddings, threshold):
        report.warn(f"[dupes] {a} and {b}: cosine similarity {sim:.3f} > {threshold}")


# --------------------------------------------------------------------------
def validate_bank(bank: Bank, skill: str | None = None, execute: bool = True, dupes: bool = False,
                  rules: Rules | None = None, embed_fn=None, timeout: float = TIMEOUT_SECONDS) -> Report:
    """Run all checks. `skill` restricts the item-level and coverage checks to one skill."""
    rules = rules or Rules()
    report = Report()
    skills = [skill] if skill else list(SKILLS)

    for path, msg in bank.load_errors.items():
        report.error(f"[load] {path}: {msg}")

    # keep only dict items; anything else is reported and skipped by the other checks
    questions = [q for q in bank.questions if isinstance(q, dict)]
    misconceptions = [m for m in bank.misconceptions if isinstance(m, dict)]
    for kind, items in (("question", bank.questions), ("misconception", bank.misconceptions)):
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                report.error(f"[schema] {kind} #{i}: not a JSON object")
    if skill:
        questions = [q for q in questions if q.get("skill") == skill]
        misconceptions = [m for m in misconceptions if m.get("skill") == skill]

    check_schema(bank, report, questions, misconceptions)
    check_ids(report, questions, misconceptions)
    check_references(bank, report, questions, misconceptions)
    check_coverage(bank, report, skills, rules)
    check_warnings(bank, report, questions)
    if execute:
        check_execution(report, questions, timeout)
    if dupes:
        check_dupes(bank, report, questions, embed_fn)

    report.info.insert(0, f"{len(questions)} questions, {len(misconceptions)} misconceptions, "
                          f"{len(bank.jobs)} jobs checked")
    return report
