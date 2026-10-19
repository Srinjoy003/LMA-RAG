r"""Validate the question bank and print a readable report. Exit code 1 if there are errors.

    .venv\Scripts\python.exe scripts\validate_bank.py                # everything
    .venv\Scripts\python.exe scripts\validate_bank.py --skill sql    # one skill
    .venv\Scripts\python.exe scripts\validate_bank.py --no-exec      # skip running read-code questions
    .venv\Scripts\python.exe scripts\validate_bank.py --dupes        # also flag near-duplicate questions
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import SKILLS, load_bank  # noqa: E402
from part1.validate import validate_bank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="project root (default: this project)")
    parser.add_argument("--skill", choices=SKILLS, help="check only this skill")
    parser.add_argument("--no-exec", action="store_true", help="do not run read-code questions")
    parser.add_argument("--dupes", action="store_true",
                        help="flag question pairs with embedding cosine similarity > 0.9 (loads the model)")
    args = parser.parse_args()

    bank = load_bank(args.root)
    report = validate_bank(bank, skill=args.skill, execute=not args.no_exec, dupes=args.dupes)

    for line in report.info:
        print(line)
    print()
    if report.errors:
        print(f"ERRORS ({len(report.errors)})")
        for msg in report.errors:
            print(f"  - {msg}")
        print()
    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)})")
        for msg in report.warnings:
            print(f"  - {msg}")
        print()
    print(f"Result: {len(report.errors)} errors, {len(report.warnings)} warnings")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
