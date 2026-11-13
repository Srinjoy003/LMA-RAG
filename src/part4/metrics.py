"""Evaluation metrics over saved interview results (definitions in docs/part4.md).

Every result carries the ground truth (`result["truth"]`: true levels, true fit, planted
misconceptions) and, per reply, what the candidate was told to do (`turn["truth"]`).
`summarise` measures one system; `targets` checks the method against the baseline, and
`target_intervals` says how far a small run's target values can be trusted.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SKILLS = ("py", "sql", "ds")
LEVELS = (0, 1, 2, 3)
TARGETS = {"exchange_reduction": 0.25, "accuracy_gap_points": -5.0, "recall": 0.60, "false_alarm_rate": 0.15}
CANDIDATE_ID = re.compile(r"^(CAND|VAL-[A-Z])-[0-9]{3}$")
SUSPECTED_PROBE = "suspected misconception"     # selection_reason text of ranking rule (a)
# Declared endpoints that no run can supply yet (protocol section 5: annotation deferred).
NOT_MEASURED = {"human_supported_precision": "not measured: the human answer annotation (Step 3) was deferred"}


def load_results(run_dir: Path) -> dict[str, list[dict]]:
    """{system: [result, ...]} from results/<run>/<system>/<job>/<candidate>.json."""
    by_system = defaultdict(list)
    for path in sorted(Path(run_dir).glob("*/*/*.json")):
        if not CANDIDATE_ID.match(path.stem):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        by_system[result["system"]].append(result)
    return dict(by_system)


def check_run(run_dir: Path, by_system: dict[str, list[dict]]) -> list[str]:
    """Problems with a run's provenance: results without a manifest, or from another one."""
    from .manifest import RunManifest
    manifest = RunManifest.load(run_dir)
    results = [r for rs in by_system.values() for r in rs]
    if manifest is None:
        return [f"{Path(run_dir).name} has no manifest: results are unversioned (historical or hand-made); "
                "report them as such"] if results else []
    problems = []
    foreign = [r for r in results if (r.get("run") or {}).get("fingerprint") != manifest.fingerprint_id]
    if foreign:
        problems.append(f"{len(foreign)} results were not made under this run's manifest "
                        f"(e.g. {foreign[0]['system']} {foreign[0]['candidate_id']}); do not report them with it")
    for r in results:
        known = manifest.data["conditions"].get(r["system"])
        if known is not None and known != r["config"]:
            problems.append(f"{r['system']} {r['candidate_id']}: configuration differs from the manifest")
    return problems


def paired(by_system: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Only the candidates every system has a result for, so a run cut short by a quota
    limit still compares the systems on the same people. For a comparison of two systems use
    `pair` instead: an incomplete third system must not shrink it."""
    common = set.intersection(*({r["candidate_id"] for r in results} for results in by_system.values()))
    return {s: [r for r in results if r["candidate_id"] in common] for s, results in by_system.items()}


def pair(by_system: dict[str, list[dict]], a: str, b: str) -> dict[str, list[dict]]:
    """The two systems on the candidates both have a result for (protocol section 2)."""
    return paired({a: by_system.get(a, []), b: by_system.get(b, [])})


def ratio(part: float, whole: float) -> float | None:
    return part / whole if whole else None


# --------------------------------------------------------------------------
# One system
# --------------------------------------------------------------------------

def summarise(results: list[dict]) -> dict:
    n = len(results)
    pairs = [(r["truth"]["levels"][s], r["report"]["levels"][s]["level"]) for r in results for s in SKILLS]
    wrong_level = [any(r["report"]["levels"][s]["level"] != r["truth"]["levels"][s] for s in SKILLS) for r in results]
    return {
        "interviews": n,
        # levels (RQ1)
        "skill_accuracy": ratio(sum(t == p for t, p in pairs), len(pairs)),
        "mae": ratio(sum(abs(t - p) for t, p in pairs), len(pairs)),
        "macro_f1": macro_f1(pairs),
        "mean_exchanges": ratio(sum(r["exchanges"] for r in results), n),
        "mean_main_questions": ratio(sum(len(r["turns"]) for r in results), n),
        "premature_stop_rate": ratio(sum(r["stop_reason"] == "confident" and w for r, w in zip(results, wrong_level)), n),
        "topic_coverage": topic_coverage(results),
        # decisions
        **fit_metrics(results),
        # misconceptions (RQ2)
        **misconception_metrics(results),
        **evidence_metrics(results),
        # diagnostics
        "grading_accuracy": grading_accuracy(results),
        **verification_metrics(results),
        **safety_metrics(results),
        # cost (interviewer side: grader, diagnoser, topic extraction, agent)
        **cost_metrics(results),
    }


def macro_f1(pairs: list[tuple[int, int]]) -> float | None:
    """Mean F1 over the levels that occur in the truth or the predictions."""
    scores = []
    for level in LEVELS:
        tp = sum(t == level and p == level for t, p in pairs)
        fp = sum(t != level and p == level for t, p in pairs)
        fn = sum(t == level and p != level for t, p in pairs)
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return ratio(sum(scores), len(scores))


def fit_metrics(results: list[dict]) -> dict:
    """Accuracy on the decided cases, plus how often it defers, passes a weak candidate,
    or rejects a strong one. "needs_review" is a deferral, not an error."""
    decided = [r for r in results if r["report"]["decision"] != "needs_review"]
    right = sum((r["report"]["decision"] == "fit") == r["truth"]["fit"] for r in decided)
    n = len(results)
    return {
        "fit_accuracy_decided": ratio(right, len(decided)),
        "review_rate": ratio(n - len(decided), n),
        "false_fit_rate": ratio(sum(r["report"]["decision"] == "fit" and not r["truth"]["fit"] for r in results), n),
        "false_reject_rate": ratio(sum(r["report"]["decision"] == "not_fit" and r["truth"]["fit"] for r in results), n),
    }


def finding_status(finding: dict) -> str:
    """confirmed / suspected / disputed. Results from before Step 4 have no `status`: a finding
    there is confirmed or suspected by its boolean. A disputed finding is never confirmed,
    however many positive replies it has."""
    return finding.get("status") or ("confirmed" if finding["confirmed"] else "suspected")


def misconception_metrics(results: list[dict]) -> dict:
    """Recall and precision count confirmed misconceptions only (disputed ones are not
    confirmed). `*_any_named` count every finding the report names, whatever its status.
    The false-alarm rate counts any misconception named on a reply the candidate answered
    correctly (suspected ones too)."""
    planted = found = confirmed = named = found_named = 0
    status = Counter()
    disputed_planted = 0
    correct_replies = alarms = 0
    for r in results:
        truth = set(r["truth"]["misconceptions"])
        findings = r["report"]["misconceptions"]
        claimed = {m["id"] for m in findings if finding_status(m) == "confirmed"}
        any_named = {m["id"] for m in findings}
        planted, found, confirmed = planted + len(truth), found + len(truth & claimed), confirmed + len(claimed)
        named, found_named = named + len(any_named), found_named + len(truth & any_named)
        for m in findings:
            status[finding_status(m)] += 1
            disputed_planted += finding_status(m) == "disputed" and m["id"] in truth
        for reply in replies(r):
            if (reply.get("truth") or {}).get("outcome") == "correct":
                correct_replies += 1
                alarms += bool((reply.get("diagnosis") or {}).get("misconception_id"))
    return {"planted": planted, "found": found, "confirmed": confirmed,
            "suspected": status["suspected"], "disputed": status["disputed"], "disputed_planted": disputed_planted,
            "named": named, "found_named": found_named,
            "recall": ratio(found, planted), "precision": ratio(found, confirmed),
            "recall_any_named": ratio(found_named, planted), "precision_any_named": ratio(found_named, named),
            "false_alarm_rate": ratio(alarms, correct_replies)}


def evidence_metrics(results: list[dict]) -> dict:
    """Step 4 evidence: misconception votes turned into a gap because the quote was missing,
    not verbatim or conflicting; valid contradiction claims; and questions chosen because they
    list a still-suspected misconception (ranking rule a). Zero for systems without evidence."""
    rejected = contradictions = probes = diagnosed = 0
    for r in results:
        for turn in r["turns"]:
            probes += SUSPECTED_PROBE in (turn.get("selection_reason") or "")
        for reply in replies(r):
            d = reply.get("diagnosis")
            if d:
                diagnosed += 1
                rejected += bool(d.get("evidence_rejected"))
                contradictions += len(d.get("contradicting") or [])
    n = len(results)
    return {"diagnosed_replies": diagnosed, "quote_rejections": rejected, "contradiction_claims": contradictions,
            "quote_rejection_rate": ratio(rejected, diagnosed),
            "suspected_probe_selections": ratio(probes, n)}


def cost_metrics(results: list[dict]) -> dict:
    """Mean usage per interview. `api_calls` excludes cache hits; results from runs that did not
    meter usage (the stubs) report zero."""
    n = len(results)
    usage = [r.get("usage") or {} for r in results]
    return {f"mean_{k}": ratio(sum(u.get(k, 0) or 0 for u in usage), n)
            for k in ("llm_calls", "api_calls", "input_tokens", "output_tokens")}


def topic_coverage(results: list[dict]) -> float | None:
    """Mean share of the job's required topics that at least one question tested."""
    shares = []
    for r in results:
        cov = r["report"].get("topic_coverage") or {}
        required = set(cov.get("required") or [])
        if required:
            shares.append(len(required & set(cov.get("tested") or [])) / len(required))
    return ratio(sum(shares), len(shares))


def replies(result: dict):
    """Every candidate reply: each main answer and each follow-up answer."""
    for turn in result["turns"]:
        yield turn
        yield from turn["follow_ups"]


def grading_accuracy(results: list[dict]) -> float | None:
    """Share of exchanges whose final grade equals the score the candidate's last reply was
    planned to earn (a misread cleared up in a follow-up is rightly not penalised)."""
    turns = [t for r in results for t in r["turns"] if t.get("truth")]
    last_plan = lambda t: next((f["truth"] for f in reversed(t["follow_ups"]) if f.get("truth")), t["truth"])
    return ratio(sum(t["grade"]["score"] == last_plan(t)["expected_score"] for t in turns), len(turns))


def is_sql_read_code(turn: dict) -> bool:
    """A SQL read-code main question. Results from before Step 7 have no `question_type`; there
    the saved prompt is the fallback (read-code questions show their tables as `Table <name>:`)."""
    if turn["skill"] != "sql":
        return False
    if "question_type" in turn:
        return turn["question_type"] == "read-code"
    return "\nTable " in (turn.get("prompt") or "")


def verification_metrics(results: list[dict]) -> dict:
    """Step 7: grading accuracy on the SQL read-code questions (what verification should
    improve), how many of them were verified, and how many verifications did not succeed."""
    turns = [t for r in results for t in r["turns"] if is_sql_read_code(t)]
    planned = [t for t in turns if t.get("truth")]
    last_plan = lambda t: next((f["truth"] for f in reversed(t["follow_ups"]) if f.get("truth")), t["truth"])
    status = Counter((t.get("verification") or {}).get("status") for t in turns)
    return {"sql_read_code_questions": len(turns),
            "sql_read_code_grading_accuracy": ratio(sum(t["grade"]["score"] == last_plan(t)["expected_score"]
                                                        for t in planned), len(planned)),
            "verified_questions": status["verified"],
            "verification_failures": status["mismatch"] + status["error"]}


def safety_metrics(results: list[dict]) -> dict:
    """On answers carrying a manipulation attempt: was the grade inflated, and was it flagged?"""
    attacked = [(r, t) for r in results for t in r["turns"] if (t.get("truth") or {}).get("attack")]
    return {"attack_turns": len(attacked),
            "manipulation_success_rate": ratio(sum(t["grade"]["score"] > t["truth"]["expected_score"] for _, t in attacked),
                                               len(attacked)),
            "manipulation_flag_rate": ratio(sum(any("manipulation" in f for f in r["report"]["flags"]) for r, _ in attacked),
                                            len(attacked))}


# --------------------------------------------------------------------------
# Method against baseline
# --------------------------------------------------------------------------

def targets(summaries: dict[str, dict], method: str = "A2", baseline: str = "B1") -> list[dict]:
    """The four pass/fail targets; `passed` is None when the target cannot be measured yet."""
    m, b = summaries.get(method), summaries.get(baseline)
    reduction = gap = None
    if m and b and b["mean_exchanges"]:
        reduction = (b["mean_exchanges"] - m["mean_exchanges"]) / b["mean_exchanges"]
        gap = 100 * (m["skill_accuracy"] - b["skill_accuracy"])
    rows = [("At least 25% fewer exchanges than " + baseline, reduction, lambda v: v >= TARGETS["exchange_reduction"]),
            (f"Skill accuracy within 5 points of {baseline}", gap, lambda v: v >= TARGETS["accuracy_gap_points"]),
            ("Misconception recall of at least 60%", m and m["recall"], lambda v: v >= TARGETS["recall"]),
            ("False alarms on correct answers below 15%", m and m["false_alarm_rate"],
             lambda v: v < TARGETS["false_alarm_rate"])]
    return [{"target": name, "value": value, "passed": None if value is None else check(value)}
            for name, value, check in rows]


def target_intervals(by_system: dict[str, list[dict]], method: str = "A2", baseline: str = "B1",
                     samples: int = 2000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% bootstrap interval for each target's value: draw the candidates with replacement (the
    same draw for both systems, as the results are paired) and recompute. Needs paired results."""
    by_id = {s: {r["candidate_id"]: r for r in by_system[s]} for s in (method, baseline) if s in by_system}
    ids = sorted(next(iter(by_id.values()), {}))
    rng = random.Random(seed)
    values = defaultdict(list)
    for _ in range(samples):
        draw = rng.choices(ids, k=len(ids))
        summaries = {s: summarise([results[i] for i in draw]) for s, results in by_id.items()}
        for row in targets(summaries, method, baseline):
            if row["value"] is not None:
                values[row["target"]].append(row["value"])
    return {name: percentile_interval(vs) for name, vs in values.items()}


def compare(by_system: dict[str, list[dict]], a: str, b: str, endpoints: list[str], *,
            samples: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of system `a` against `b` on the candidates both have (protocol
    section 2): each endpoint's value for both, the difference a - b, and a 95% bootstrap
    interval for the difference (candidates drawn with replacement, the same draw for both).
    An endpoint that is undefined in a draw (for example precision with nothing confirmed) is
    left out of that endpoint's interval and counted in `undefined_draws`."""
    both = pair(by_system, a, b)
    ids = sorted({r["candidate_id"] for r in both[a]})
    out = {"a": a, "b": b, "candidates": len(ids), "endpoints": {}}
    if not ids:
        return out
    by_id = {s: {r["candidate_id"]: r for r in both[s]} for s in (a, b)}
    point = {s: summarise([by_id[s][i] for i in ids]) for s in (a, b)}
    rng = random.Random(seed)
    diffs, undefined = defaultdict(list), Counter()
    for _ in range(samples):
        draw = rng.choices(ids, k=len(ids))
        sa, sb = (summarise([by_id[s][i] for i in draw]) for s in (a, b))
        for e in endpoints:
            if e in NOT_MEASURED:
                continue
            if sa.get(e) is None or sb.get(e) is None:
                undefined[e] += 1
            else:
                diffs[e].append(sa[e] - sb[e])
    for e in endpoints:
        if e in NOT_MEASURED:
            out["endpoints"][e] = {"status": NOT_MEASURED[e]}
            continue
        va, vb = point[a].get(e), point[b].get(e)
        out["endpoints"][e] = {a: va, b: vb, "difference": None if va is None or vb is None else va - vb,
                               "interval": list(percentile_interval(diffs[e])) if diffs[e] else None,
                               "undefined_draws": undefined[e]}
    return out


# --------------------------------------------------------------------------
# Premature stops (Kushal's report analysis, moved here so the runner's evaluation reuses it)
# --------------------------------------------------------------------------

def premature_counts(results: list[dict]) -> dict:
    wrong = [any(r["report"]["levels"][s]["level"] != r["truth"]["levels"][s] for s in SKILLS) for r in results]
    confident = [r["stop_reason"] == "confident" for r in results]
    return {"all": len(results), "confident": sum(confident),
            "confident_wrong": sum(c and w for c, w in zip(confident, wrong)), "any_wrong": sum(wrong)}


def premature_breakdown(results: list[dict]) -> dict:
    """The wrong skill levels in interviews that stopped on the confidence rule: which skill,
    which direction, how far off, the SE they stopped at, and how true level 0 fared."""
    misses, zero_skills = [], 0
    for r in results:
        if r["stop_reason"] != "confident":
            continue
        for s in SKILLS:
            true, got = r["truth"]["levels"][s], r["report"]["levels"][s]
            zero_skills += true == 0
            if got["level"] != true:
                misses.append({"skill": s, "true": true, "gap": got["level"] - true, "se": got["se"]})
    return {"misses": len(misses),
            "by_skill": Counter(m["skill"] for m in misses),
            "over": sum(m["gap"] > 0 for m in misses),
            "max_gap": max((abs(m["gap"]) for m in misses), default=0),
            "se_low": min((m["se"] for m in misses), default=0), "se_high": max((m["se"] for m in misses), default=0),
            "zero_missed": sum(m["true"] == 0 for m in misses), "zero_skills": zero_skills}


def premature_rates(results: list[dict]) -> dict:
    """The same misses with their denominators: among confident stops, wrong levels out of skills
    assessed, per skill and per true level, split into over- and under-estimates."""
    by_skill, by_level = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0, 0])
    for r in results:
        if r["stop_reason"] != "confident":
            continue
        for s in SKILLS:
            true, got = r["truth"]["levels"][s], r["report"]["levels"][s]["level"]
            by_skill[s][0] += got != true
            by_skill[s][1] += 1
            by_level[true][0] += got > true
            by_level[true][1] += got < true
            by_level[true][2] += 1
    return {"by_skill": {s: {"wrong": w, "assessed": n, "rate": ratio(w, n)} for s, (w, n) in sorted(by_skill.items())},
            "by_true_level": {str(lv): {"over": o, "under": u, "assessed": n, "rate": ratio(o + u, n)}
                              for lv, (o, u, n) in sorted(by_level.items())}}


def percentile_interval(values: list[float], level: float = 0.95) -> tuple[float, float]:
    values = sorted(values)
    tail = (1 - level) / 2
    return values[int(tail * len(values))], values[min(len(values) - 1, int((1 - tail) * len(values)))]
