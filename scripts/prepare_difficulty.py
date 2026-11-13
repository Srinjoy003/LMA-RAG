"""Prepare the Step 3 blind difficulty check without printing the answer key."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part4.difficulty import prepare_difficulty  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="bank repository root")
    parser.add_argument("--out", type=Path, default=ROOT / "results/annotation/difficulty-v1")
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--per-skill", type=int, default=7)
    args = parser.parse_args()
    try:
        result = prepare_difficulty(args.root, args.out, seed=args.seed, per_skill=args.per_skill)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(f"Prepared {result['items']} items ({result['per_skill']} per skill).")
    print(f"Output: {result['output']}")
    print("Distribute packet.md and each rater's own CSV only. Private files are for the preparer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
