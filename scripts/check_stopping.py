"""Audit Step 6 validation runs; never use final evaluation candidates for tuning.

Run interviews with run_experiment.py first (see docs/step6_stopping.md).
This reads saved results and their manifest; it makes no model calls.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part4.manifest import RunManifest, file_sha256  # noqa: E402
from part4.metrics import (SKILLS, check_run, compare, load_results, premature_counts,
                           premature_rates, ratio, summarise)  # noqa: E402

ENDPOINTS = ["skill_accuracy", "mae", "mean_exchanges", "premature_stop_rate",
             "false_fit_rate", "false_reject_rate", "review_rate", "recall", "precision"]


def audit(results):
    """All stop reasons enter level accuracy; confident-only rates keep their denominators."""
    by_level = {str(level): {"wrong": 0, "assessed": 0, "over": 0, "under": 0}
                for level in range(4)}
    missing, low_mass, boundary_errors = 0, 0, 0
    for result in results:
        threshold = result["config"].get("level_probability_threshold", .7)
        for skill in SKILLS:
            row = result["report"]["levels"][skill]
            true, got = result["truth"]["levels"][skill], row["level"]
            counts = by_level[str(true)]
            counts["assessed"] += 1
            counts["wrong"] += got != true
            counts["over"] += got > true
            counts["under"] += got < true
            probability = row.get("level_probability")
            missing += probability is None
            low_mass += probability is not None and probability < threshold
            boundary_errors += (probability is not None and probability < threshold and got != true)
    for counts in by_level.values():
        counts["error_rate"] = ratio(counts["wrong"], counts["assessed"])
    return {"stop_reasons": dict(Counter(r["stop_reason"] for r in results)),
            "all_stops_by_true_level": by_level,
            "skills_missing_probability": missing,
            "skills_below_probability_threshold": low_mass,
            "wrong_skills_below_probability_threshold": boundary_errors,
            "confident_interviews": premature_counts(results),
            "confident_skills": premature_rates(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="folder under results/")
    parser.add_argument("--json", type=Path, required=True, help="new output file (will not overwrite)")
    args = parser.parse_args()
    run_dir = ROOT / "results" / args.run
    by_system = load_results(run_dir)
    manifest = RunManifest.load(run_dir)
    if not manifest or not by_system:
        parser.error("a non-empty, versioned validation run is required")
    problems = check_run(run_dir, by_system)
    for system, results in by_system.items():
        ids = [r["candidate_id"] for r in results]
        if any(not cid.startswith("VAL-P-") for cid in ids):
            problems.append(f"{system}: only separate VAL-P pilot candidates are allowed here")
        if len(set(ids)) != len(ids) or any(r.get("errors") for r in results):
            problems.append(f"{system}: duplicate candidate or interview errors; inspect the run")
    if problems:
        parser.error("; ".join(problems))
    if args.json.exists():
        parser.error(f"{args.json} exists; choose a new output path")
    payload = {"purpose": "Step 6 validation only; stub outcomes are not live evidence",
               "run": args.run, "fingerprint_id": manifest.fingerprint_id,
               "analysis_sha256": file_sha256(Path(__file__)),
               "setup": manifest.data["fingerprint"]["setup"],
               "conditions": manifest.data["conditions"],
               "candidates": manifest.data["candidates"],
               "systems": {s: {"summary": summarise(rs), "stopping": audit(rs)}
                           for s, rs in by_system.items()},
               "comparisons": [compare(by_system, a, b, ENDPOINTS)
                               for a, b in [("A2EB", "A2E"), ("A2B", "A2")]
                               if a in by_system and b in by_system]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for system, row in payload["systems"].items():
        summary = row["summary"]
        print(f"{system}: accuracy={summary['skill_accuracy']:.1%}, "
              f"exchanges={summary['mean_exchanges']:.2f}, "
              f"stops={row['stopping']['stop_reasons']}")
    print(f"Saved {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
