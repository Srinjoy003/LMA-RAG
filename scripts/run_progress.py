r"""Show how far a run has got, as progress bars. Reads only the saved results.

    .venv/bin/python scripts/run_progress.py                      # once
    .venv/bin/python scripts/run_progress.py --watch              # refresh every 30 s (Ctrl-C to quit)
    .venv/bin/python scripts/run_progress.py --run reviewed-main-v1 --systems B1 A2E A4 ADAPT12 A5E

A candidate counts as complete once every system has a saved result for it.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 40


def bar(done: int, total: int) -> str:
    filled = round(WIDTH * done / total) if total else 0
    return f"[{'#' * filled}{'.' * (WIDTH - filled)}] {done:>3}/{total} ({100 * done / total if total else 0:5.1f}%)"


def saved(run_dir: Path, system: str) -> set[str]:
    return {p.stem for p in (run_dir / system).glob("*/*.json")}


def failures(run_dir: Path) -> int:
    path = run_dir / "failures.jsonl"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0


def last_log_line(log: Path) -> str:
    if not log.exists():
        return "(no log yet)"
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1][:100] if lines else "(empty log)"


def show(run: str, systems: list[str], candidates: list[str]) -> None:
    run_dir = ROOT / "results" / run
    done = {s: saved(run_dir, s) & set(candidates) for s in systems}
    complete = [c for c in candidates if all(c in done[s] for s in systems)]
    total = len(systems) * len(candidates)
    print(f"Run {run}   {time.strftime('%H:%M:%S')}\n")
    print(f"{'All interviews':<20}{bar(sum(len(d) for d in done.values()), total)}")
    print(f"{'Complete candidates':<20}{bar(len(complete), len(candidates))}")
    print(f"{'Full blocks of 12':<20}{len(complete) // 12} of {len(candidates) // 12}\n")
    for s in systems:
        print(f"  {s:<18}{bar(len(done[s]), len(candidates))}")
    print(f"\nFailed attempts logged: {failures(run_dir)}")
    print(f"Last log line: {last_log_line(ROOT / 'results' / f'{run}.log')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="reviewed-main-v2")
    parser.add_argument("--systems", nargs="+", default=["B1", "A2E", "A4", "ADAPT12", "A5E"])
    parser.add_argument("--order", default=str(ROOT / "docs" / "final_run_order.txt"),
                        help="file with the run's candidate ids, one per line")
    parser.add_argument("--watch", action="store_true", help="refresh every 30 seconds")
    args = parser.parse_args()
    candidates = [c.strip() for c in Path(args.order).read_text(encoding="utf-8").splitlines() if c.strip()]
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")      # clear the terminal
        show(args.run, args.systems, candidates)
        if not args.watch:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
