r"""Compute the evaluation metrics for one experiment run (see src/part4/metrics.py, docs/part4.md).

    .venv/bin/python scripts/evaluate.py --run stub-stub
    .venv/bin/python scripts/evaluate.py --run rv1-main --comparisons --json results/rv1-main/metrics.json

The table compares the systems on the candidates they all have results for. `--comparisons`
also runs the paired comparisons declared in docs/experiment_matrix.json (C1-C6) whose two
conditions are in the run, each on the candidates those two share, with a 95% bootstrap
interval for every difference. The run's manifest is checked first; results from another
manifest are reported, and exclusions (pairs that failed MAX_ATTEMPTS times) are listed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part4.manifest import failure_counts  # noqa: E402
from part4.metrics import (  # noqa: E402
    check_run, compare, load_results, paired, premature_rates, summarise, target_intervals, targets,
)
from part4.runner import MAX_ATTEMPTS, RESULTS_DIR  # noqa: E402

MATRIX = ROOT / "docs" / "experiment_matrix.json"

ROWS = [  # (label, metric, format)
    ("Interviews", "interviews", "{:.0f}"),
    ("Skill accuracy", "skill_accuracy", "{:.1%}"),
    ("Mean level error (MAE)", "mae", "{:.2f}"),
    ("Macro-F1 over levels", "macro_f1", "{:.2f}"),
    ("Exchanges per interview", "mean_exchanges", "{:.1f}"),
    ("Main questions per interview", "mean_main_questions", "{:.1f}"),
    ("Premature stops", "premature_stop_rate", "{:.1%}"),
    ("Required topics tested", "topic_coverage", "{:.1%}"),
    ("Fit accuracy (decided cases)", "fit_accuracy_decided", "{:.1%}"),
    ("Sent to human review", "review_rate", "{:.1%}"),
    ("Not-fit passed as fit", "false_fit_rate", "{:.1%}"),
    ("Fit rejected", "false_reject_rate", "{:.1%}"),
    ("Misconception recall", "recall", "{:.1%}"),
    ("Misconception precision", "precision", "{:.1%}"),
    ("Recall, any named", "recall_any_named", "{:.1%}"),
    ("Precision, any named", "precision_any_named", "{:.1%}"),
    ("Confirmed findings", "confirmed", "{:.0f}"),
    ("Suspected findings", "suspected", "{:.0f}"),
    ("Disputed findings", "disputed", "{:.0f}"),
    ("False alarms on correct answers", "false_alarm_rate", "{:.1%}"),
    ("Quote-rejected diagnoses", "quote_rejections", "{:.0f}"),
    ("Contradiction claims", "contradiction_claims", "{:.0f}"),
    ("Suspected-probe questions / interview", "suspected_probe_selections", "{:.2f}"),
    ("Grades matching the plan", "grading_accuracy", "{:.1%}"),
    ("  on SQL read-code questions", "sql_read_code_grading_accuracy", "{:.1%}"),
    ("SQL read-code questions verified", "verified_questions", "{:.0f}"),
    ("SQL verification failures", "verification_failures", "{:.0f}"),
    ("Manipulation raised the grade", "manipulation_success_rate", "{:.1%}"),
    ("Manipulation flagged", "manipulation_flag_rate", "{:.1%}"),
    ("LLM calls / interview", "mean_llm_calls", "{:.1f}"),
    ("API calls (uncached) / interview", "mean_api_calls", "{:.1f}"),
]


def show(value, fmt: str) -> str:
    return "-" if value is None else fmt.format(value)


def show_target(value, target: str) -> str:
    return f"{value:+.1f} points" if "points" in target else f"{value:.1%}"


def declared_comparisons(systems: set[str]) -> list[dict]:
    """The matrix's paired comparisons whose two conditions both have results in this run."""
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    preset = {name: c["preset"] for name, c in matrix["conditions"].items() if c.get("preset")}
    out = []
    for c in matrix["comparisons"]:
        if not c.get("paired", True):
            continue
        a, b = (preset.get(name) for name in c["conditions"])
        if a in systems and b in systems:
            out.append({**c, "systems": [a, b]})
    return out


def show_comparison(result: dict, spec: dict) -> None:
    a, b = result["a"], result["b"]
    print(f"\n{spec['id']} ({spec['question']}): {a} vs {b}, {result['candidates']} paired candidates")
    for name, e in result["endpoints"].items():
        if "status" in e:
            print(f"  {name}: {e['status']}")
            continue
        fmt = (lambda v: "-" if v is None else f"{v:.3f}")
        interval = "-" if e["interval"] is None else f"{e['interval'][0]:+.3f} to {e['interval'][1]:+.3f}"
        undefined = f", {e['undefined_draws']} undefined draws" if e["undefined_draws"] else ""
        diff = "-" if e["difference"] is None else f"{e['difference']:+.3f}"
        print(f"  {name}: {a} {fmt(e[a])}, {b} {fmt(e[b])}, difference {diff} (95% CI {interval}{undefined})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="stub-stub", help="results/<run>/ folder")
    parser.add_argument("--method", default="A2")
    parser.add_argument("--baseline", default="B1")
    parser.add_argument("--comparisons", action="store_true", help="also run the declared paired comparisons")
    parser.add_argument("--samples", type=int, default=2000, help="bootstrap draws (protocol: 2000)")
    parser.add_argument("--json", type=Path, help="also write all numbers to this file")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run
    all_results = load_results(run_dir)
    if not all_results:
        parser.error(f"no results in {run_dir}: run scripts/run_experiment.py first")
    provenance = check_run(run_dir, all_results)
    for problem in provenance:
        print(f"Provenance: {problem}")
    have = {(s, r["candidate_id"]) for s, rs in all_results.items() for r in rs}
    excluded = sorted(k for k, n in failure_counts(run_dir).items() if n >= MAX_ATTEMPTS and k not in have)
    for system, cid in excluded:
        print(f"Excluded: {system} {cid} (failed {MAX_ATTEMPTS} times)")
    if provenance or excluded:
        print()

    by_system = paired(all_results)
    left_out = sum(len(rs) for rs in all_results.values()) - sum(len(rs) for rs in by_system.values())
    if left_out:
        print(f"Left out {left_out} results whose candidate is missing from another system "
              "(the paired comparisons below use each pair's own candidates).\n")
    systems = sorted(by_system, key=lambda s: (s != args.baseline, s))
    summaries = {s: summarise(by_system[s]) for s in systems}

    width = max(len(label) for label, _, _ in ROWS) + 2
    print("".ljust(width) + "".join(s.rjust(9) for s in systems))
    for label, key, fmt in ROWS:
        print(label.ljust(width) + "".join(show(summaries[s][key], fmt).rjust(9) for s in systems))
    m = summaries.get(args.method, {})
    if m:
        print(f"\n{args.method}: {m['found']} of {m['planted']} planted misconceptions confirmed, {m['confirmed']} "
              f"confirmed in all, {m['disputed']} disputed ({m['disputed_planted']} of them planted)")

    rows = []
    if args.method in summaries and args.baseline in summaries:
        print(f"\nTargets ({args.method} against {args.baseline})")
        rows = targets(summaries, args.method, args.baseline)
        intervals = target_intervals(by_system, args.method, args.baseline, samples=args.samples)
        for row in rows:
            mark = {True: "PASS", False: "FAIL", None: "n/a "}[row["passed"]]
            value = "-" if row["value"] is None else show_target(row["value"], row["target"])
            if row["target"] in intervals:
                low, high = intervals[row["target"]]
                row["interval"] = [low, high]
                value += f"  (95% CI {show_target(low, row['target'])} to {show_target(high, row['target'])})"
            print(f"  {mark}  {row['target']}: {value}")

    premature = {s: premature_rates(by_system[s]) for s in systems}
    for s in systems:
        rates = premature[s]
        if rates["by_skill"]:
            per_skill = ", ".join(f"{k} {v['wrong']}/{v['assessed']}" for k, v in rates["by_skill"].items())
            per_level = ", ".join(f"true {k}: {v['over']} over, {v['under']} under of {v['assessed']}"
                                  for k, v in rates["by_true_level"].items())
            print(f"\nWrong levels after confident stops, {s}: {per_skill}; {per_level}")

    comparisons = []
    if args.comparisons:
        for spec in declared_comparisons(set(all_results)):
            endpoints = spec["endpoints"] + spec.get("secondary", [])
            result = compare(all_results, *spec["systems"], endpoints, samples=args.samples)
            show_comparison(result, spec)
            comparisons.append({"id": spec["id"], **result})
        if not comparisons:
            print("\nNo declared comparison has both of its conditions in this run.")

    if args.json:
        args.json.write_text(json.dumps({"summaries": summaries, "targets": rows, "premature": premature,
                                         "comparisons": comparisons, "provenance": provenance,
                                         "excluded": [list(k) for k in excluded]}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
