r"""Run the systems on the simulated candidates and save every result (see src/part4/runner.py).

    .venv/bin/python scripts/run_experiment.py                                   # stubs, B1 A2 A4 A5, all 60
    .venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --run main
    .venv/bin/python scripts/run_experiment.py --grader gemini --simulator ollama --candidates 3 --systems A2

A balanced subset: the first 12 candidates (or 24, 36, ...) cover each job x even/uneven x
fit/not-fit once, so `--candidates 12 --systems B1 A2` is the smallest fair comparison.

Sharing the work across accounts: put each extra key in .env with a suffix (GROQ_API_KEY_2,
GEMINI_API_KEY_2, ...) and start one run per account, each on its share of the candidates:
    ... --run subset --share 1/3
    ... --run subset --share 2/3 --keys 2
    ... --run subset --share 3/3 --keys 3

Explicit candidates (protocol section 3): `--ids CAND-001 CAND-004`, `--ids-file FILE` (one id per
line), or `--allocation nilkanta` / `--allocation souradeep` for the balanced final-run halves in
docs/experiment_matrix.json. `--candidates-dir` selects another population, such as the pilot
candidates in data/candidates_validation/pilot (ids VAL-P-001 ...).

Results go to results/<run>/<system>/<job>/<candidate>.json (default run name: <grader>-<simulator>).
Finished interviews are skipped, so a run stopped by a quota limit continues where it left off.
The run's manifest.json records the code, bank, prompts, setup, configurations and candidates;
a run made with anything different, or an old folder without a manifest, is refused before any
interview: choose a new --run name. Then: scripts/evaluate.py --run <run>.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.providers import use_key_set  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part4.candidates import CANDIDATES_DIR, load_candidates  # noqa: E402
from part4.manifest import RunManifestError  # noqa: E402
from part4.runner import RESULTS_DIR, Setup, run_experiment  # noqa: E402
from shared.interfaces import SYSTEMS  # noqa: E402

MATRIX = ROOT / "docs" / "experiment_matrix.json"


def chosen_ids(args, parser) -> list[str] | None:
    given = [x for x in (args.ids, args.ids_file, args.allocation) if x]
    if len(given) > 1:
        parser.error("use only one of --ids, --ids-file and --allocation")
    if args.ids:
        return args.ids
    if args.ids_file:
        return [line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    if args.allocation:
        allocation = json.loads(MATRIX.read_text(encoding="utf-8"))["allocation"]
        if args.allocation not in allocation or not isinstance(allocation[args.allocation], list):
            names = [k for k, v in allocation.items() if isinstance(v, list)]
            parser.error(f"--allocation must be one of {', '.join(names)}")
        return allocation[args.allocation]
    return None

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
    parser.add_argument("--ids", nargs="+", metavar="ID", help="exactly these candidate ids")
    parser.add_argument("--ids-file", type=Path, help="file with one candidate id per line")
    parser.add_argument("--allocation", help="a named half of the final run from docs/experiment_matrix.json")
    parser.add_argument("--candidates-dir", type=Path, default=CANDIDATES_DIR,
                        help="folder of candidate keys (default: the 60 evaluation candidates)")
    parser.add_argument("--share", metavar="K/N", help="only the K-th of N equal shares of the candidates")
    parser.add_argument("--keys", metavar="NAME", help="use the keys ending in _NAME in .env, e.g. GROQ_API_KEY_2, GEMINI_API_KEY_2")
    parser.add_argument("--run", help="results folder name (default: <grader>-<simulator>)")
    parser.add_argument("--operator", help="who is running this session (recorded in the run manifest)")
    parser.add_argument("--rerun", action="store_true", help="redo interviews that already have a result")
    args = parser.parse_args()

    ids = chosen_ids(args, parser)
    try:
        candidates = load_candidates(args.candidates_dir, ids)
    except (KeyError, ValueError) as e:
        parser.error(str(e).strip("'\""))
    if not candidates:
        parser.error(f"no candidates in {args.candidates_dir}: run scripts/make_candidates.py first")
    if args.candidates and ids:
        parser.error("--candidates cannot be combined with explicit ids")
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
    try:
        counts = run_experiment(args.systems, candidates, jobs, setup, run_dir, rerun=args.rerun,
                                operator=args.operator, share=args.share or args.allocation)
    except RunManifestError as e:
        parser.exit(1, "Refused, nothing was run:\n" + "".join(f"  - {p}\n" for p in e.problems))
    print(f"\nDone {counts['done']}, already there {counts['skipped']}, failed {counts['failed']}, "
          f"excluded {counts['excluded']}.")
    print(f"Next: .venv/bin/python scripts/evaluate.py --run {run_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
