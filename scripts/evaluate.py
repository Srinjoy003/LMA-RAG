r"""Compute the evaluation metrics for one experiment run (see src/part4/metrics.py, docs/part4.md).

    .venv/bin/python scripts/evaluate.py --run stub-stub
    .venv/bin/python scripts/evaluate.py --run main --json results/main/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part4.metrics import load_results, summarise, targets  # noqa: E402
from part4.runner import RESULTS_DIR  # noqa: E402

ROWS = [  # (label, metric, format)
    ("Interviews", "interviews", "{:.0f}"),
    ("Skill accuracy", "skill_accuracy", "{:.1%}"),
    ("Mean level error (MAE)", "mae", "{:.2f}"),
    ("Macro-F1 over levels", "macro_f1", "{:.2f}"),
    ("Exchanges per interview", "mean_exchanges", "{:.1f}"),
    ("Main questions per interview", "mean_main_questions", "{:.1f}"),
    ("Premature stops", "premature_stop_rate", "{:.1%}"),
    ("Fit accuracy (decided cases)", "fit_accuracy_decided", "{:.1%}"),
    ("Sent to human review", "review_rate", "{:.1%}"),
    ("Not-fit passed as fit", "false_fit_rate", "{:.1%}"),
    ("Fit rejected", "false_reject_rate", "{:.1%}"),
    ("Misconception recall", "recall", "{:.1%}"),
    ("Misconception precision", "precision", "{:.1%}"),
    ("False alarms on correct answers", "false_alarm_rate", "{:.1%}"),
    ("Grades matching the plan", "grading_accuracy", "{:.1%}"),
    ("Manipulation raised the grade", "manipulation_success_rate", "{:.1%}"),
    ("Manipulation flagged", "manipulation_flag_rate", "{:.1%}"),
]


def show(value, fmt: str) -> str:
    return "-" if value is None else fmt.format(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="stub-stub", help="results/<run>/ folder")
    parser.add_argument("--method", default="A2")
    parser.add_argument("--baseline", default="B1")
    parser.add_argument("--json", type=Path, help="also write all numbers to this file")
    args = parser.parse_args()

    by_system = load_results(RESULTS_DIR / args.run)
    if not by_system:
        parser.error(f"no results in {RESULTS_DIR / args.run}: run scripts/run_experiment.py first")
    systems = sorted(by_system, key=lambda s: (s != args.baseline, s))
    summaries = {s: summarise(by_system[s]) for s in systems}

    width = max(len(label) for label, _, _ in ROWS) + 2
    print("".ljust(width) + "".join(s.rjust(9) for s in systems))
    for label, key, fmt in ROWS:
        print(label.ljust(width) + "".join(show(summaries[s][key], fmt).rjust(9) for s in systems))
    m = summaries.get(args.method, {})
    if m:
        print(f"\n{args.method}: {m['found']} of {m['planted']} planted misconceptions confirmed, {m['confirmed']} confirmed in all")

    print(f"\nTargets ({args.method} against {args.baseline})")
    rows = targets(summaries, args.method, args.baseline)
    for row in rows:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[row["passed"]]
        value = "-" if row["value"] is None else (f"{row['value']:+.1f} points" if "points" in row["target"]
                                                  else f"{row['value']:.1%}")
        print(f"  {mark}  {row['target']}: {value}")

    if args.json:
        args.json.write_text(json.dumps({"summaries": summaries, "targets": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
