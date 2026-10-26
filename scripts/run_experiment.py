r"""Run the systems on the simulated candidates and save every result (see src/part4/runner.py).

    .venv/bin/python scripts/run_experiment.py                                   # stubs, B1 A2 A4 A5, all 60
    .venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --run main
    .venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --candidates 3 --systems A2

A balanced subset: the first 12 candidates (or 24, 36, ...) cover each job x even/uneven x
fit/not-fit once, so `--candidates 12 --systems B1 A2` is the smallest fair comparison.

Sharing the work across accounts: put each extra key in .env with a suffix (GROQ_API_KEY_2,
LLM_API_KEY_2, ...) and start one run per account, each on its share of the candidates:
    ... --run subset --share 1/3
    ... --run subset --share 2/3 --keys 2
    ... --run subset --share 3/3 --keys 3

Results go to results/<run>/<system>/<job>/<candidate>.json (default run name: <grader>-<simulator>).
Finished interviews are skipped, so a run stopped by a quota limit continues where it left off.
Then: scripts/evaluate.py --run <run>.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.providers import use_key_set  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part4.candidates import load_candidates  # noqa: E402
from part4.runner import RESULTS_DIR, Setup, run_experiment  # noqa: E402
from shared.interfaces import SYSTEMS  # noqa: E402

CHOICES = ["stub", "gemini", "groq", "ollama"]


def share(text: str, parser) -> tuple[int, int]:
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        parser.error(f"--share takes K/N, such as 2/3, not {text!r}")
    if not 1 <= k <= n:
        parser.error(f"--share {text}: K must be between 1 and N")
    return k, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=["B1", "A2", "A4", "A5"], choices=sorted(SYSTEMS))
    parser.add_argument("--grader", default="stub", choices=CHOICES, help="provider for grading and diagnosis")
    parser.add_argument("--simulator", default="stub", choices=CHOICES, help="provider for the candidates' answers")
    parser.add_argument("--simulator-model", help="override the simulator provider's default model")
    parser.add_argument("--candidates", type=int, help="only the first N candidates (default: all)")
    parser.add_argument("--share", metavar="K/N", help="only the K-th of N equal shares of the candidates")
    parser.add_argument("--keys", metavar="NAME", help="use the keys ending in _NAME in .env, e.g. GROQ_API_KEY_2")
    parser.add_argument("--run", help="results folder name (default: <grader>-<simulator>)")
    parser.add_argument("--rerun", action="store_true", help="redo interviews that already have a result")
    args = parser.parse_args()

    candidates = load_candidates()
    if not candidates:
        parser.error("no candidates in data/candidates/: run scripts/make_candidates.py first")
    candidates = candidates[:args.candidates] if args.candidates else candidates
    if args.share:
        k, n = share(args.share, parser)
        candidates = candidates[k - 1::n]
    setup = Setup(grader=args.grader, simulator=args.simulator, simulator_model=args.simulator_model)
    if setup.problems():
        parser.error("; ".join(setup.problems()))
    if args.keys:
        live = [p for p in (args.grader, args.simulator) if p != "stub"]
        switched = use_key_set(args.keys, live)
        if not switched:
            parser.error(f"no key ending in _{args.keys} in .env for {' or '.join(live) or 'the stubs'}")
        print(f"Using key set {args.keys}: {', '.join(switched)}")
    run_dir = RESULTS_DIR / (args.run or f"{args.grader}-{args.simulator}")
    jobs = {j["id"]: j for j in load_bank(ROOT).jobs}

    print(f"{len(args.systems)} systems x {len(candidates)} candidates -> {run_dir}")
    counts = run_experiment(args.systems, candidates, jobs, setup, run_dir, rerun=args.rerun)
    print(f"\nDone {counts['done']}, already there {counts['skipped']}, failed {counts['failed']}.")
    print(f"Next: .venv/bin/python scripts/evaluate.py --run {run_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
