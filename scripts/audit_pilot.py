"""Step 8 completeness, provenance and direct A2E/A2 comparison; no model calls."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part4.manifest import RunManifest, failure_counts, file_sha256  # noqa: E402
from part4.metrics import check_run, compare, load_results, premature_rates, summarise  # noqa: E402


def completeness(by_system, plan):
    expected = {(s, cid) for s in plan["systems"] for cid in plan["candidate_ids"]}
    ids = [(s, r["candidate_id"]) for s, rs in by_system.items() for r in rs]
    counts = Counter(ids)
    return {"expected": len(expected), "saved": len(ids),
            "missing": sorted(expected - set(ids)), "unexpected": sorted(set(ids) - expected),
            "duplicates": sorted(k for k, n in counts.items() if n > 1)}


def elapsed_seconds(result):
    """Whole-interview wall time, including simulator and any waits; not API latency."""
    return (datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00")) -
            datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="rv4-groq-pilot-20261114")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error("output already exists; choose a new JSON filename")
    plan_path = ROOT / "docs/step8_pilot_plan.json"
    plan = json.loads(plan_path.read_text())
    run_dir = ROOT / "results" / args.run
    manifest = RunManifest.load(run_dir)
    by_system = load_results(run_dir)
    if not manifest or not by_system:
        parser.error("no versioned results yet; live pilot has not produced saved interviews")
    problems = check_run(run_dir, by_system)
    if any(not r["candidate_id"].startswith("VAL-P-") for rs in by_system.values() for r in rs):
        problems.append("final evaluation candidates are not allowed in the pilot")
    if manifest.data["fingerprint"]["setup"] != plan["setup"]:
        problems.append("provider/model setup differs from the declared live pilot")
    for system, config in manifest.data["conditions"].items():
        if system in plan["systems"] and config.get("models", {}).get("grader") != plan["grader_model"]:
            problems.append(f"{system}: grader model differs from the declared pilot")
    if problems:
        parser.error("; ".join(problems))
    inventory = completeness(by_system, plan)
    errors = [{"system": s, "candidate": r["candidate_id"], "errors": r["errors"]}
              for s, rs in by_system.items() for r in rs if r.get("errors")]
    failures = [{"system": s, "candidate": cid, "attempts": n}
                for (s, cid), n in sorted(failure_counts(run_dir).items())]
    complete = not any(inventory[k] for k in ("missing", "unexpected", "duplicates"))
    report = {"owner": "souradeep", "run": args.run, "plan_sha256": file_sha256(plan_path),
              "fingerprint_id": manifest.fingerprint_id, "inventory": inventory,
              "complete": complete, "interview_errors": errors, "failed_attempts": failures,
              "status": "ready for transcript review" if complete and not errors else "incomplete or flagged; inspect before conclusions",
              "systems": {s: {"summary": summarise(rs), "premature_rates": premature_rates(rs),
                               "stop_reasons": dict(Counter(r["stop_reason"] for r in rs)),
                               "wall_seconds": [elapsed_seconds(r) for r in rs]}
                          for s, rs in by_system.items()},
              "A2E_vs_A2": compare(by_system, "A2E", "A2", plan["additional_comparison"]["endpoints"])
                           if {"A2E", "A2"} <= set(by_system) else None,
              "human_supported_precision": "not measured: Step 3 deferred",
              "limitations": plan["limits"]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{inventory['saved']}/{inventory['expected']} saved; {len(inventory['missing'])} missing; {report['status']}")
    print(f"Saved {args.json}")
    return 0 if complete and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
