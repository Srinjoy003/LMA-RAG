r"""Make the final report's figures and numbers from the final run.

    .venv/bin/python report/make_final_figures.py
    .venv/bin/python report/make_final_figures.py --run reviewed-main-v2-final22

Writes report/figures/final_*.pdf and report/final_numbers.tex (one \newcommand per number the
final report quotes, all starting with \f), so the report never carries a hand-copied number.
Only candidates with a result in every system are used, so every comparison is paired.
Intervals are 95% percentile intervals from 2000 bootstrap draws of candidates (seed 0).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report"))

from make_figures import GRID, INK, INK_MUTED, GOOD, CRITICAL, NEUTRAL, bootstrap, grid_lines  # noqa: E402
from part4.metrics import compare, load_results, paired, summarise  # noqa: E402
from part4.runner import RESULTS_DIR  # noqa: E402

OUT = ROOT / "report"
FIGURES = OUT / "figures"
SKILLS = ("py", "sql", "ds")
SYSTEMS = ("B1", "ADAPT12", "A2E", "A5E", "A4")       # the order used everywhere
NAMES = {"B1": "Fixed (B1)", "ADAPT12": "Adaptive, 12 questions (ADAPT12)", "A2E": "Full system (A2E)",
         "A5E": "Job-blind (A5E)", "A4": "No theory of mind (A4)"}
# Categorical slots 1-5 of the reference palette, in fixed order (validated: CVD and normal-vision
# separation pass; three slots are below 3:1 contrast, so every figure also names the systems).
COLOR = {"B1": "#2a78d6", "A2E": "#eb6834", "A4": "#1baf7a", "ADAPT12": "#eda100", "A5E": "#e87ba4"}
DIAGNOSING = ("A2E", "A5E", "A4")                     # the systems that name misconceptions
COMPARISONS = {   # macro name: (a, b, endpoints); C1-C4 of docs/experiment_matrix.json
    "FullVsFixed": ("A2E", "B1", ["skill_accuracy", "mean_exchanges", "mae"]),          # C1, RQ1
    "AdaptVsFixed": ("ADAPT12", "B1", ["skill_accuracy", "mae"]),                       # C2, RQ1 equal length
    "FullVsAfour": ("A2E", "A4", ["recall", "precision", "false_alarm_rate", "skill_accuracy",
                                  "mean_exchanges"]),                                    # C3, RQ2
    "BlindVsFull": ("A5E", "A2E", ["skill_accuracy", "recall", "topic_coverage"]),      # C4, retrieval
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.pdf")


def metric(name: str):
    """A statistic over a list of results, for bootstrapping; None when undefined."""
    return lambda results: summarise(results)[name]


def interval(by_system: dict, name: str, systems=SYSTEMS) -> dict:
    return bootstrap({s: by_system[s] for s in systems}, lambda rs: _defined(metric(name)(rs)))


def _defined(value):
    return float("nan") if value is None else value


def nan_interval(low_high):
    low, high = low_high
    return (None if np.isnan(low) else low, None if np.isnan(high) else high)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_accuracy(by_system: dict, summary: dict, ci: dict) -> None:
    """Skill levels judged right, one row per system, with its 95% interval."""
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    rows = list(reversed(SYSTEMS))
    for y, s in enumerate(rows):
        mean, (low, high) = 100 * summary[s]["skill_accuracy"], ci[s]
        ax.plot([100 * low, 100 * high], [y, y], color=COLOR[s], linewidth=2, solid_capstyle="round")
        ax.scatter([mean], [y], s=46, color=COLOR[s], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(100 * high + 1.5, y, f"{mean:.0f}%", va="center", color=INK, fontsize=8.5)
    ax.axvline(100 * summary["B1"]["skill_accuracy"], color=NEUTRAL, linewidth=0.9, linestyle=(0, (3, 3)),
               zorder=0)
    ax.set_yticks(range(len(rows)), [NAMES[s] for s in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(20, 100)
    ax.set_xlabel("Skill levels judged correctly (%), with 95% interval; dashed line: fixed baseline")
    grid_lines(ax, "x")
    save(fig, "final_accuracy")


def fig_beliefs(summary: dict, ci_recall: dict, ci_precision: dict) -> None:
    """RQ2: how many planted misconceptions each system confirms (recall) against how many of
    its confirmed findings are planted ones (precision)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for s in DIAGNOSING:
        x, y = 100 * summary[s]["recall"], 100 * summary[s]["precision"]
        (xl, xh), (yl, yh) = ci_recall[s], ci_precision[s]
        ax.plot([100 * xl, 100 * xh], [y, y], color=COLOR[s], linewidth=1.6, alpha=0.8)
        ax.plot([x, x], [100 * yl, 100 * yh], color=COLOR[s], linewidth=1.6, alpha=0.8)
        ax.scatter([x], [y], s=60, color=COLOR[s], edgecolor="white", linewidth=1.5, zorder=3)
        left = s == "A2E"          # A2E and A5E sit close together: label them on opposite sides
        ax.annotate(NAMES[s], (x, y), xytext=(-8 if left else 8, 8 if left else -12),
                    textcoords="offset points", ha="right" if left else "left", fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Planted misconceptions confirmed (recall, %)")
    ax.set_ylabel("Confirmed findings that were planted\n(precision, %)")
    grid_lines(ax, "both")
    save(fig, "final_beliefs")


def fig_length(by_system: dict) -> dict:
    """Mean exchanges per interview: main questions plus follow-ups."""
    fig, ax = plt.subplots(figsize=(6.0, 2.1))
    rows = list(reversed(SYSTEMS))
    lengths = {}
    for y, s in enumerate(rows):
        main = np.mean([len(r["turns"]) for r in by_system[s]])
        follow = np.mean([r["exchanges"] - len(r["turns"]) for r in by_system[s]])
        lengths[s] = (main, follow)
        ax.barh(y, main, height=0.6, color=COLOR[s])
        ax.text(main / 2, y, f"{main:.1f}", ha="center", va="center", color=INK, fontsize=8.5)
        if follow > 0.05:
            ax.barh(y, follow, left=main, height=0.6, color=COLOR[s], alpha=0.4, edgecolor="white", linewidth=2)
            ax.text(main + follow + 0.4, y, f"+{follow:.1f}", va="center", color=INK_MUTED, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [NAMES[s] for s in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(m + f for m, f in lengths.values()) + 3)
    ax.set_xlabel("Mean exchanges per interview (solid: main questions, light: follow-ups)")
    grid_lines(ax, "x")
    save(fig, "final_length")
    return lengths


def verdict_counts(results: list[dict]) -> dict:
    decided = [r for r in results if r["report"]["decision"] != "needs_review"]
    right = sum((r["report"]["decision"] == "fit") == r["truth"]["fit"] for r in decided)
    return {"right": right, "review": len(results) - len(decided), "wrong": len(decided) - right}


def fig_verdicts(by_system: dict) -> dict:
    """Fit / not-fit verdicts: right, sent to a human, or wrong."""
    segments = [("right", "Right verdict", GOOD), ("review", "Sent to a human", NEUTRAL),
                ("wrong", "Wrong verdict", CRITICAL)]
    fig, ax = plt.subplots(figsize=(6.0, 2.1))
    rows = list(reversed(SYSTEMS))
    counts = {s: verdict_counts(by_system[s]) for s in SYSTEMS}
    n = len(by_system[SYSTEMS[0]])
    for y, s in enumerate(rows):
        left = 0
        for key, label, color in segments:
            k = counts[s][key]
            if k:
                ax.barh(y, k, left=left, height=0.6, color=color, edgecolor="white", linewidth=2,
                        label=label if y == len(rows) - 1 or not ax.get_legend_handles_labels()[1].count(label)
                        else None)
                ax.text(left + k / 2, y, str(k), ha="center", va="center", fontsize=8.5,
                        color=INK if key == "review" else "white", fontweight="bold")
            left += k
    handles, labels = ax.get_legend_handles_labels()
    keep = {l: h for h, l in zip(handles, labels)}
    ax.legend([keep[l] for _, l, _ in segments if l in keep], [l for _, l, _ in segments if l in keep],
              frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_yticks(range(len(rows)), [NAMES[s] for s in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, n)
    ax.set_xlabel(f"Interviews ({n} candidates)")
    save(fig, "final_verdicts")
    return counts


# --------------------------------------------------------------------------
# Numbers for the text
# --------------------------------------------------------------------------

def pct(x, digits: int = 0) -> str:
    return "--" if x is None else f"{100 * x:.{digits}f}\\%"


def points(x) -> str:
    return f"{100 * x:+.1f}"


def level_errors(results: list[dict]) -> dict:
    """Wrong skill levels: too high or too low, overall and after a confident stop, plus the
    confident-stop errors by skill and the true-level-0 skills misjudged."""
    out = {"over": 0, "under": 0, "confident_stops": 0, "conf_wrong": 0, "conf_skills": 0,
           "zero": 0, "zero_wrong": 0, "conf_two_off": 0, **{f"conf_{s}": 0 for s in SKILLS}}
    for r in results:
        confident = r["stop_reason"] == "confident"
        out["confident_stops"] += confident
        for s in SKILLS:
            true, got = r["truth"]["levels"][s], r["report"]["levels"][s]["level"]
            out["over"] += got > true
            out["under"] += got < true
            if confident:
                out["conf_skills"] += 1
                out["conf_wrong"] += got != true
                out[f"conf_{s}"] += got != true
                out["conf_two_off"] += abs(got - true) >= 2
                out["zero"] += true == 0
                out["zero_wrong"] += true == 0 and got != true
    return out


def held_belief_questions(results: list[dict]) -> float:
    """Main questions per interview that the candidate answered from a planted misconception."""
    return sum((t.get("truth") or {}).get("outcome") == "misconception"
               for r in results for t in r["turns"]) / len(results)


def tex_name(system: str) -> str:
    return {"B1": "BOne", "ADAPT12": "Adapt", "A2E": "Full", "A5E": "Blind", "A4": "Afour"}[system]


def write_numbers(values: dict) -> None:
    lines = ["% Generated by report/make_final_figures.py; do not edit by hand."]
    for name, value in values.items():
        value = str(value)
        value = "$-$" + value[1:] if value.startswith("-") else value
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (OUT / "final_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "final_numbers.tex")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="reviewed-main-v2-final22")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    by_system = paired({s: rs for s, rs in load_results(RESULTS_DIR / args.run).items() if s in SYSTEMS})
    n = len(by_system["B1"])
    summary = {s: summarise(by_system[s]) for s in SYSTEMS}
    ci_acc = interval(by_system, "skill_accuracy")
    ci_recall = {s: nan_interval(v) for s, v in interval(by_system, "recall", DIAGNOSING).items()}
    ci_precision = {s: nan_interval(v) for s, v in interval(by_system, "precision", DIAGNOSING).items()}

    fig_accuracy(by_system, summary, ci_acc)
    fig_beliefs(summary, ci_recall, ci_precision)
    lengths = fig_length(by_system)
    verdicts = fig_verdicts(by_system)

    v = {"fN": n, "fInterviews": n * len(SYSTEMS),
         "fPlanted": summary["A2E"]["planted"]}
    for s in SYSTEMS:
        t, m = tex_name(s), summary[s]
        v.update({f"fAcc{t}": pct(m["skill_accuracy"]), f"fAccLow{t}": pct(ci_acc[s][0]),
                  f"fAccHigh{t}": pct(ci_acc[s][1]), f"fMae{t}": f"{m['mae']:.2f}",
                  f"fExch{t}": f"{m['mean_exchanges']:.1f}", f"fMain{t}": f"{lengths[s][0]:.1f}",
                  f"fFollow{t}": f"{lengths[s][1]:.1f}", f"fPremature{t}": pct(m["premature_stop_rate"]),
                  f"fCoverage{t}": pct(m["topic_coverage"]), f"fReview{t}": pct(m["review_rate"]),
                  f"fRight{t}": verdicts[s]["right"], f"fWrong{t}": verdicts[s]["wrong"],
                  f"fReviewN{t}": verdicts[s]["review"], f"fCalls{t}": f"{m['mean_llm_calls']:.0f}",
                  f"fGradingMatch{t}": pct(m["grading_accuracy"])})
    for s in SYSTEMS:
        t, e = tex_name(s), level_errors(by_system[s])
        v.update({f"fOver{t}": e["over"], f"fUnder{t}": e["under"], f"fConfStops{t}": e["confident_stops"],
                  f"fConfWrong{t}": e["conf_wrong"], f"fConfSkills{t}": e["conf_skills"],
                  f"fConfWrongPy{t}": e["conf_py"], f"fConfWrongSql{t}": e["conf_sql"],
                  f"fConfWrongDs{t}": e["conf_ds"], f"fZeroWrong{t}": e["zero_wrong"], f"fZero{t}": e["zero"],
                  f"fConfTwoOff{t}": e["conf_two_off"],
                  f"fHeldBelief{t}": f"{held_belief_questions(by_system[s]):.1f}"})
    for s in DIAGNOSING:
        t, m = tex_name(s), summary[s]
        v.update({f"fRecall{t}": pct(m["recall"]), f"fPrecision{t}": pct(m["precision"]),
                  f"fRecallLow{t}": pct(ci_recall[s][0]), f"fRecallHigh{t}": pct(ci_recall[s][1]),
                  f"fPrecisionLow{t}": pct(ci_precision[s][0]), f"fPrecisionHigh{t}": pct(ci_precision[s][1]),
                  f"fFound{t}": m["found"], f"fConfirmed{t}": m["confirmed"],
                  f"fSuspected{t}": m["suspected"], f"fDisputed{t}": m["disputed"],
                  f"fFalseAlarm{t}": pct(m["false_alarm_rate"], 1)})

    for name, (a, b, endpoints) in COMPARISONS.items():
        result = compare(by_system, a, b, endpoints)
        for e, r in result["endpoints"].items():
            key = "".join(w.capitalize() for w in e.split("_"))
            scale = (lambda x: f"{x:+.1f}") if e in ("mean_exchanges",) else \
                    (lambda x: f"{x:+.2f}") if e == "mae" else points
            v[f"f{name}{key}Diff"] = scale(r["difference"])
            v[f"f{name}{key}Low"], v[f"f{name}{key}High"] = (scale(x) for x in r["interval"])
    write_numbers(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
