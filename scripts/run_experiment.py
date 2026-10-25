r"""Run the systems on the simulated candidates and save every result (see src/part4/runner.py).

    .venv/bin/python scripts/run_experiment.py                                   # stubs, B1 A2 A4 A5, all 60
    .venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --run main
    .venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --candidates 3 --systems A2

Results go to results/<run>/<system>/<job>/<candidate>.json (default run name: <grader>-<simulator>).
Finished interviews are skipped, so a run stopped by a quota limit continues where it left off.
Then: scripts/evaluate.py --run <run>.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import load_bank  # noqa: E402
from part4.candidates import load_candidates  # noqa: E402
from part4.runner import RESULTS_DIR, Setup, run_experiment  # noqa: E402
from shared.interfaces import SYSTEMS  # noqa: E402

CHOICES = ["stub", "gemini", "groq", "ollama"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=["B1", "A2", "A4", "A5"], choices=sorted(SYSTEMS))
    parser.add_argument("--grader", default="stub", choices=CHOICES, help="provider for grading and diagnosis")
    parser.add_argument("--simulator", default="stub", choices=CHOICES, help="provider for the candidates' answers")
    parser.add_argument("--simulator-model", help="override the simulator provider's default model")
    parser.add_argument("--candidates", type=int, help="only the first N candidates (default: all)")
    parser.add_argument("--run", help="results folder name (default: <grader>-<simulator>)")
    parser.add_argument("--rerun", action="store_true", help="redo interviews that already have a result")
    args = parser.parse_args()

    candidates = load_candidates()
    if not candidates:
        parser.error("no candidates in data/candidates/: run scripts/make_candidates.py first")
    candidates = candidates[:args.candidates] if args.candidates else candidates
    setup = Setup(grader=args.grader, simulator=args.simulator, simulator_model=args.simulator_model)
    if setup.problems():
        parser.error("; ".join(setup.problems()))
    run_dir = RESULTS_DIR / (args.run or f"{args.grader}-{args.simulator}")
    jobs = {j["id"]: j for j in load_bank(ROOT).jobs}

    print(f"{len(args.systems)} systems x {len(candidates)} candidates -> {run_dir}")
    counts = run_experiment(args.systems, candidates, jobs, setup, run_dir, rerun=args.rerun)
    print(f"\nDone {counts['done']}, already there {counts['skipped']}, failed {counts['failed']}.")
    print(f"Next: .venv/bin/python scripts/evaluate.py --run {run_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
