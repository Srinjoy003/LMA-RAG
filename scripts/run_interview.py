r"""Run one interview with the stubs (rule-based candidate, tag grader and diagnoser) and print a summary.

    .venv\Scripts\python.exe scripts\run_interview.py --system A2 --seed 0
    .venv\Scripts\python.exe scripts\run_interview.py --system B1 --candidate examples\candidate.example.json --save

The engine never sees the candidate key. As a convenience for this dry run only, the
respondent's hidden answer plans are merged into the result as `truth` fields afterwards,
the way the experiment runner will.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.run import run_interview, save_result  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part1.index import DEFAULT_INDEX_DIR, QuestionIndex  # noqa: E402
from part4.runner import merge_truths  # noqa: E402
from shared.interfaces import SYSTEMS, Components  # noqa: E402
from shared.stubs import RuleBasedRespondent, TagDiagnoser, TagGrader  # noqa: E402

QUESTION_WIDTH = 100


def get_index(bank) -> QuestionIndex:
    if (DEFAULT_INDEX_DIR / "meta.json").exists():
        return QuestionIndex.load(DEFAULT_INDEX_DIR, bank=bank)
    print("No saved index found; building one in memory (loads the embedding model)...")
    return QuestionIndex.build(bank)


def print_summary(result: dict, candidate: dict) -> None:
    print(f"\n=== {result['system']} | job {result['job_id']} | candidate {result['candidate_id']} "
          f"| seed {result['config']['seed']} ===")
    for turn in result["turns"]:
        grade = turn["grade"]
        question = turn["prompt"].split("\n")[0]
        if len(question) > QUESTION_WIDTH:
            question = question[:QUESTION_WIDTH - 3] + "..."
        diagnosis = turn["diagnosis"]
        text = f"{diagnosis['cause']}" + (f" ({diagnosis['misconception_id']})" if diagnosis.get("misconception_id") else "") \
            if diagnosis else "-"
        print(f"{turn['index']:>2}. [{turn['skill']} L{turn['level']}] {turn['question_id']}  {question}")
        print(f"      grade {grade['score']} (votes {grade['votes']}, weight {turn['evidence_weight']:.2f})"
              f"  diagnosis: {text}  follow-ups: {len(turn['follow_ups'])}")
        for follow_up in turn["follow_ups"]:
            cause = follow_up.get("diagnosis", {}).get("cause", "not diagnosed")
            print(f"        follow-up: {follow_up['prompt']}  -> {cause}")

    report = result["report"]
    print("\nScorecard")
    for skill, row in report["levels"].items():
        meets = {True: "meets", False: "below", None: "uncertain"}[row["meets_minimum"]]
        print(f"  {skill:<3} level {row['level']}  estimate {row['estimate']:+.2f}  SE {row['se']:.2f}"
              f"  confident {str(row['confident']):<5}  minimum {row['minimum']}  -> {meets}")
    print(f"\nDecision: {report['decision']} - {report['decision_reason']}")
    print(f"Stop reason: {result['stop_reason']}; main questions {len(result['turns'])}; "
          f"exchanges {result['exchanges']}")

    found = {m["id"]: m["confirmed"] for m in report["misconceptions"]}
    planted = candidate.get("misconceptions", [])
    print("Misconceptions confirmed:", sorted(m for m, c in found.items() if c) or "none")
    print("Misconceptions suspected:", sorted(m for m, c in found.items() if not c) or "none")
    print("Planted in the candidate:", planted or "none",
          "| found:", [m for m in planted if found.get(m)] or "none")
    if report["flags"]:
        print("Flags:", report["flags"])
    if result["errors"]:
        print("Errors:", result["errors"])
    covered = report["topic_coverage"]
    print(f"Required topics tested: {len(covered['tested'])} of {len(covered['required'])}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="A2", choices=sorted(SYSTEMS), help="system preset (default A2)")
    parser.add_argument("--job", default=None, help="job id from data/jobs.json (default: the candidate's job)")
    parser.add_argument("--candidate", type=Path, default=ROOT / "examples" / "candidate.example.json",
                        help="candidate key JSON")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="write results/<system>/<job>/<candidate>.json")
    args = parser.parse_args(argv)

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    bank = load_bank()
    job = bank.get_job(args.job or candidate["job_id"])
    if job is None:
        parser.error(f"unknown job {args.job or candidate['job_id']!r}")

    config = SYSTEMS[args.system].with_(seed=args.seed)
    components = Components(question_source=get_index(bank), grader=TagGrader(votes=config.grader_votes),
                            diagnoser=TagDiagnoser(), misconceptions=bank.misconception_by_id)
    respondent = RuleBasedRespondent(candidate)
    result = run_interview(job, respondent, config, components, candidate_id=candidate["id"],
                           rng=random.Random(args.seed))
    merge_truths(result, respondent.truths)

    print_summary(result, candidate)
    if args.save:
        print("Saved", save_result(result, ROOT))
    return result


if __name__ == "__main__":
    main()
