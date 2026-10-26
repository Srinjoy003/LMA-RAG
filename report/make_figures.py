r"""Make the report's figures and numbers from one experiment run (default: midsub).

    .venv/bin/python report/make_figures.py
    .venv/bin/python report/make_figures.py --run midsub --skip-sweep

Writes report/figures/*.pdf and report/numbers.tex (one \newcommand per number the text
quotes), so the report never carries a hand-copied number. The stopping sweep reruns the
engine on stub candidates (seeds 777-779, not the evaluation set); --skip-sweep reuses
nothing and leaves that figure out.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.factory import build_components  # noqa: E402
from engine.run import run_interview  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part4.candidates import make_candidates, truth_for  # noqa: E402
from part4.metrics import load_results, paired, percentile_interval, replies, summarise, target_intervals  # noqa: E402
from part4.runner import RESULTS_DIR  # noqa: E402
from shared.interfaces import SYSTEMS  # noqa: E402
from shared.stubs import RuleBasedRespondent  # noqa: E402

OUT = ROOT / "report"
FIGURES = OUT / "figures"
SKILLS = ("py", "sql", "ds")
SKILL_NAMES = {"py": "Python", "sql": "SQL", "ds": "Data structures"}
NAMES = {"B1": "Fixed (B1)", "A2": "Adaptive (A2)"}

# Validated categorical slots 1-3 and the status steps (dataviz reference palette)
COLOR = {"B1": "#2a78d6", "A2": "#eb6834", "third": "#1baf7a"}
GOOD, CRITICAL, SERIOUS, NEUTRAL = "#0ca30c", "#d03b3b", "#ec835a", "#b9b8b3"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def bootstrap(results_by_system: dict, statistic, samples: int = 2000, seed: int = 0) -> dict:
    """{system: (low, high)}: 95% interval of `statistic(results)`, drawing candidates with
    replacement (the same draw for every system, as the results are paired)."""
    by_id = {s: {r["candidate_id"]: r for r in rs} for s, rs in results_by_system.items()}
    ids = sorted(next(iter(by_id.values())))
    rng = random.Random(seed)
    values = {s: [] for s in by_id}
    for _ in range(samples):
        draw = rng.choices(ids, k=len(ids))
        for s, results in by_id.items():
            values[s].append(statistic([results[i] for i in draw]))
    return {s: percentile_interval(v) for s, v in values.items()}


def skill_accuracy(results: list[dict], skills=SKILLS) -> float:
    pairs = [(r["truth"]["levels"][s], r["report"]["levels"][s]["level"]) for r in results for s in skills]
    return sum(t == p for t, p in pairs) / len(pairs)


def grid_lines(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.pdf")


# --------------------------------------------------------------------------
# Figures from the experiment run
# --------------------------------------------------------------------------

def fig_accuracy(by_system: dict) -> None:
    """Share of skill levels judged right, per skill and overall, with 95% intervals."""
    groups = [(SKILL_NAMES[s], (s,)) for s in SKILLS] + [("All skills", SKILLS)]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    width, x = 0.36, np.arange(len(groups))
    for offset, system in ((-0.5, "B1"), (0.5, "A2")):
        means, lows, highs = [], [], []
        for _, skills in groups:
            mean = skill_accuracy(by_system[system], skills)
            low, high = bootstrap({system: by_system[system]}, lambda rs: skill_accuracy(rs, skills))[system]
            means.append(100 * mean)
            lows.append(100 * (mean - low))
            highs.append(100 * (high - mean))
        bars = ax.bar(x + offset * (width + 0.02), means, width, color=COLOR[system], label=NAMES[system],
                      yerr=[lows, highs], error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": INK_MUTED})
        last = bars[-1]
        ax.text(last.get_x() + last.get_width() / 2, 3, f"{means[-1]:.0f}%", ha="center", va="bottom",
                color="white", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Levels judged correctly (%)")
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    grid_lines(ax)
    save(fig, "accuracy")


def fig_confusion(by_system: dict) -> None:
    """True level against judged level, all skills pooled, one panel per system."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(BLUES)
    counts = {}
    for system in ("B1", "A2"):
        m = np.zeros((4, 4), dtype=int)
        for r in by_system[system]:
            for s in SKILLS:
                m[r["truth"]["levels"][s], r["report"]["levels"][s]["level"]] += 1
        counts[system] = m
    top = max(m.max() for m in counts.values())
    for ax, system in zip(axes, ("B1", "A2")):
        m = counts[system]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=top)
        for i in range(4):
            for j in range(4):
                if m[i, j]:
                    ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8.5,
                            color="white" if m[i, j] > top * 0.55 else INK)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel("Judged level")
        ax.set_ylabel("True level")
        ax.set_title(f"{NAMES[system]}: {np.trace(m)} of {m.sum()} right")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    save(fig, "confusion")
    return counts


def verdict_counts(results: list[dict]) -> dict:
    right = sum(r["report"]["decision"] != "needs_review" and (r["report"]["decision"] == "fit") == r["truth"]["fit"]
                for r in results)
    review = sum(r["report"]["decision"] == "needs_review" for r in results)
    return {"right": right, "review": review, "wrong": len(results) - right - review}


def fig_verdicts(by_system: dict) -> None:
    """Fit / not-fit verdicts: right, deferred to a human, or wrong."""
    segments = [("right", "Right verdict", GOOD), ("review", "Sent to a human", NEUTRAL),
                ("wrong", "Wrong verdict", CRITICAL)]
    fig, ax = plt.subplots(figsize=(6.0, 1.45))
    for row, system in enumerate(("A2", "B1")):
        counts, left = verdict_counts(by_system[system]), 0
        for key, label, color in segments:
            n = counts[key]
            if n:
                ax.barh(row, n, left=left, height=0.55, color=color, edgecolor="white", linewidth=2,
                        label=label if row == 0 else None)
                ax.text(left + n / 2, row, str(n), ha="center", va="center", fontsize=8.5,
                        color=INK if key == "review" else "white", fontweight="bold")
            left += n
    ax.set_yticks([0, 1], [NAMES["A2"], NAMES["B1"]])
    ax.set_xlabel("Interviews (24 candidates)")
    ax.set_xlim(0, 24)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "verdicts")


def fig_length(by_system: dict) -> None:
    """How long the interviews were: main questions plus follow-ups."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.12})
    for row, system in enumerate(("B1", "A2")):
        results = by_system[system]
        main = np.mean([len(r["turns"]) for r in results])
        follow = np.mean([r["exchanges"] - len(r["turns"]) for r in results])
        left.barh(row, main, height=0.55, color=COLOR[system])
        left.barh(row, follow, left=main, height=0.55, color=COLOR[system], alpha=0.45, edgecolor="white",
                  linewidth=2)
        left.text(main / 2, row, f"{main:.1f}", ha="center", va="center", color="white", fontweight="bold")
        if follow > 0.5:
            left.text(main + follow / 2, row, f"{follow:.1f}", ha="center", va="center", color=INK)
        exchanges = [r["exchanges"] for r in results]
        jitter = np.random.default_rng(row).uniform(-0.15, 0.15, len(exchanges))
        right.scatter(exchanges, row + jitter, s=18, color=COLOR[system], edgecolor="white", linewidth=0.6,
                      zorder=3)
        right.plot([np.mean(exchanges)] * 2, [row - 0.3, row + 0.3], color=INK, linewidth=1.5, zorder=4)
    for ax in (left, right):
        ax.set_yticks([0, 1], [NAMES["B1"], NAMES["A2"]])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        grid_lines(ax, "x")
    left.set_xlabel("Mean exchanges\n(dark: main questions, light: follow-ups)")
    right.set_xlabel("Exchanges in each interview\n(dot: one interview, line: mean)")
    right.set_yticklabels([])
    save(fig, "length")


def misconception_breakdown(results: list[dict]) -> dict:
    """Planted misconceptions found and missed, and confirmed ones that were not planted,
    split by whether a reply the candidate answered correctly was among their evidence."""
    out = Counter()
    for r in results:
        planted = set(r["truth"]["misconceptions"])
        confirmed = {m["id"] for m in r["report"]["misconceptions"] if m["confirmed"]}
        out["found"] += len(planted & confirmed)
        out["missed"] += len(planted - confirmed)
        for mid in confirmed - planted:
            outcomes = {(reply.get("truth") or {}).get("outcome") for reply in replies(r)
                        if (reply.get("diagnosis") or {}).get("misconception_id") == mid}
            out["extra_on_correct" if "correct" in outcomes else "extra_on_wrong"] += 1
    return dict(out)


def fig_misconceptions(results: list[dict]) -> dict:
    counts = misconception_breakdown(results)
    rows = [("Planted, confirmed", counts.get("found", 0), GOOD),
            ("Planted, missed", counts.get("missed", 0), NEUTRAL),
            ("Not planted, from wrong or partial answers", counts.get("extra_on_wrong", 0), SERIOUS),
            ("Not planted, involving a correct answer", counts.get("extra_on_correct", 0), CRITICAL)]
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    for i, (label, n, color) in enumerate(reversed(rows)):
        ax.barh(i, n, height=0.6, color=color)
        ax.text(n + 1, i, str(n), va="center", color=INK)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Misconceptions (adaptive system A2, 24 interviews)")
    grid_lines(ax, "x")
    save(fig, "misconceptions")
    return counts


def fig_trajectory(results: list[dict], candidate_id: str) -> None:
    """One interview: each skill's uncertainty after every main question."""
    r = next(x for x in results if x["candidate_id"] == candidate_id)
    threshold = r["config"]["uncertainty_threshold"]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    colors = {"py": COLOR["B1"], "sql": COLOR["A2"], "ds": COLOR["third"]}
    for skill in SKILLS:
        se = [1.0] + [t["scorecard_after"][skill]["se"] for t in r["turns"]]
        x = range(len(se))
        ax.plot(x, se, color=colors[skill], linewidth=2, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8, label=SKILL_NAMES[skill])
    ax.axhline(threshold, color=INK_MUTED, linewidth=1)
    ax.text(0.1, threshold - 0.035, f"stop once every skill is below {threshold}", va="top", color=INK_MUTED,
            fontsize=8)
    ax.set_xlabel("Main questions asked")
    ax.set_ylabel("Uncertainty (SE, logits)")
    ax.set_xlim(0, len(r["turns"]) + 0.5)
    ax.set_ylim(0.3, 1.05)
    ax.legend(frameon=False, loc="upper right")
    grid_lines(ax)
    save(fig, "trajectory")


# --------------------------------------------------------------------------
# Stopping sweep on stub candidates (not the evaluation set)
# --------------------------------------------------------------------------

def stub_run(config, bank, candidates, jobs) -> dict:
    components = build_components(config, live=False, root=ROOT)
    results = []
    for c in candidates:
        job = jobs[c["job_id"]]
        r = run_interview(job, RuleBasedRespondent(c), config, components, candidate_id=c["id"],
                          rng=random.Random(config.seed))
        r["truth"] = truth_for(c, job)
        results.append(r)
    return summarise(results)


def fig_stopping(bank) -> dict:
    jobs = {j["id"]: j for j in bank.jobs}
    candidates = [c for seed in (777, 778, 779) for c in make_candidates(bank, n=60, seed=seed)]
    baseline = stub_run(SYSTEMS["B1"], bank, candidates, jobs)
    points = {thr: stub_run(SYSTEMS["A2"].with_(uncertainty_threshold=thr), bank, candidates, jobs)
              for thr in (0.7, 0.65, 0.6, 0.55)}
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    xs = [p["mean_exchanges"] for p in points.values()]
    ys = [100 * p["skill_accuracy"] for p in points.values()]
    ax.plot(xs, ys, color=COLOR["A2"], linewidth=2, marker="o", markersize=6, markeredgecolor="white",
            label=NAMES["A2"] + ", stopping threshold varied")
    for thr, x, y in zip(points, xs, ys):
        ax.annotate(f"SE < {thr}" + ("  (chosen)" if thr == 0.6 else ""), (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK,
                    fontweight="bold" if thr == 0.6 else "normal")
    ax.scatter([baseline["mean_exchanges"]], [100 * baseline["skill_accuracy"]], s=45, color=COLOR["B1"],
               edgecolor="white", zorder=3, label=NAMES["B1"])
    ax.set_xlabel("Exchanges per interview")
    ax.set_ylabel("Levels judged correctly (%)")
    ax.set_ylim(60, 90)
    ax.legend(frameon=False, loc="lower right")
    grid_lines(ax)
    save(fig, "stopping")
    return {"baseline": baseline, "points": points}


# --------------------------------------------------------------------------
# Numbers for the text
# --------------------------------------------------------------------------

def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}\\%"


def write_numbers(values: dict) -> None:
    lines = ["% Generated by report/make_figures.py; do not edit by hand."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "numbers.tex")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="midsub")
    parser.add_argument("--example", default="CAND-013", help="candidate for the uncertainty figure")
    parser.add_argument("--skip-sweep", action="store_true", help="leave out the stub stopping sweep (slow)")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    by_system = paired(load_results(RESULTS_DIR / args.run))
    by_system = {s: by_system[s] for s in ("B1", "A2")}
    s = {name: summarise(rs) for name, rs in by_system.items()}
    b1, a2 = s["B1"], s["A2"]
    intervals = target_intervals(by_system, "A2", "B1")
    gap_low, gap_high = intervals["Skill accuracy within 5 points of B1"]
    acc_ci = bootstrap(by_system, skill_accuracy)

    fig_accuracy(by_system)
    counts = fig_confusion(by_system)
    fig_verdicts(by_system)
    fig_length(by_system)
    mis = fig_misconceptions(by_system["A2"])
    fig_trajectory(by_system["A2"], args.example)

    bank = load_bank(ROOT)
    verdicts = {name: verdict_counts(rs) for name, rs in by_system.items()}
    over = {name: sum(int(counts[name][i, j]) for i in range(4) for j in range(4) if j > i) for name in counts}
    under = {name: sum(int(counts[name][i, j]) for i in range(4) for j in range(4) if j < i) for name in counts}
    values = {
        "nCandidates": len(by_system["A2"]),
        "nQuestions": len(bank.questions), "nMisconceptions": len(bank.misconceptions), "nJobs": len(bank.jobs),
        "accBase": pct(b1["skill_accuracy"], 1), "accOurs": pct(a2["skill_accuracy"], 1),
        "accBaseLow": pct(acc_ci["B1"][0]), "accBaseHigh": pct(acc_ci["B1"][1]),
        "accOursLow": pct(acc_ci["A2"][0]), "accOursHigh": pct(acc_ci["A2"][1]),
        "accGap": f"{100 * (a2['skill_accuracy'] - b1['skill_accuracy']):.1f}",
        "accGapLow": f"{gap_low:.0f}", "accGapHigh": f"{gap_high:.0f}",
        "maeBase": f"{b1['mae']:.2f}", "maeOurs": f"{a2['mae']:.2f}",
        "fOneBase": f"{b1['macro_f1']:.2f}", "fOneOurs": f"{a2['macro_f1']:.2f}",
        "overBase": over["B1"], "overOurs": over["A2"], "underBase": under["B1"], "underOurs": under["A2"],
        "exchBase": f"{b1['mean_exchanges']:.1f}", "exchOurs": f"{a2['mean_exchanges']:.1f}",
        "mainOurs": f"{a2['mean_main_questions']:.1f}",
        "followOurs": f"{a2['mean_exchanges'] - a2['mean_main_questions']:.1f}",
        "fitAccBase": pct(b1["fit_accuracy_decided"]), "fitAccOurs": pct(a2["fit_accuracy_decided"]),
        "reviewBase": pct(b1["review_rate"]), "reviewOurs": pct(a2["review_rate"]),
        "rightBase": verdicts["B1"]["right"], "rightOurs": verdicts["A2"]["right"],
        "wrongBase": verdicts["B1"]["wrong"], "wrongOurs": verdicts["A2"]["wrong"],
        "prematureOurs": pct(a2["premature_stop_rate"]),
        "planted": a2["planted"], "found": a2["found"], "confirmedAll": a2["confirmed"],
        "recallOurs": pct(a2["recall"]), "precisionOurs": pct(a2["precision"]),
        "extraWrong": mis.get("extra_on_wrong", 0), "extraCorrect": mis.get("extra_on_correct", 0),
        "falseAlarmOurs": f"{100 * a2['false_alarm_rate']:.1f}\\%",
        "gradingBase": pct(b1["grading_accuracy"]), "gradingOurs": pct(a2["grading_accuracy"]),
        "attackTurns": a2["attack_turns"], "manipulationOurs": pct(a2["manipulation_success_rate"] or 0),
        "lengthChange": f"{100 * (a2['mean_exchanges'] / b1['mean_exchanges'] - 1):.0f}",
        "recallLow": pct(intervals["Misconception recall of at least 60%"][0]),
        "recallHigh": pct(intervals["Misconception recall of at least 60%"][1]),
        "alarmHigh": f"{100 * intervals['False alarms on correct answers below 15%'][1]:.1f}\\%",
        "callsBase": f"{np.mean([r['usage']['llm_calls'] for r in by_system['B1']]):.0f}",
        "callsOurs": f"{np.mean([r['usage']['llm_calls'] for r in by_system['A2']]):.0f}",
    }
    if not args.skip_sweep:
        sweep = fig_stopping(bank)
        chosen, quick = sweep["points"][0.6], sweep["points"][0.7]
        values.update({
            "stubAccBase": pct(sweep["baseline"]["skill_accuracy"]),
            "stubAccOurs": pct(chosen["skill_accuracy"]), "stubExchOurs": f"{chosen['mean_exchanges']:.1f}",
            "stubAccQuick": pct(quick["skill_accuracy"]), "stubExchQuick": f"{quick['mean_exchanges']:.1f}",
            "stubAccLong": pct(sweep["points"][0.55]["skill_accuracy"]),
            "stubExchLong": f"{sweep['points'][0.55]['mean_exchanges']:.1f}",
        })
    write_numbers(values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
