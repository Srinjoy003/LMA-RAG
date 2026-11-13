r"""Generate simulated candidates' hidden keys (see src/part4/candidates.py).

    .venv/bin/python scripts/make_candidates.py --prefix VAL-P --seed 3102 --n 12   # pilot population
    .venv/bin/python scripts/make_candidates.py --prefix VAL-A --seed 3101 --n 12   # annotation source

Deterministic: the same --prefix, --n, --seed and bank always give the same candidates.

The 60 evaluation candidates in data/candidates/ (prefix CAND, seed 2026) are frozen for the
experiment (docs/evaluation_baseline.json hashes every file). This script never replaces
existing candidate files: it refuses a folder that already holds any, unless --replace is
given, and replacing the evaluation set also needs --i-know-this-breaks-the-baseline.
Validation populations go to data/candidates_validation/<pilot|annotation|...> by default.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import load_bank  # noqa: E402
from part4.candidates import CANDIDATES_DIR, candidate_files, make_candidates, save_candidates, truth_for  # noqa: E402

VALIDATION_DIR = ROOT / "data" / "candidates_validation"
DEFAULT_FOLDERS = {"CAND": CANDIDATES_DIR, "VAL-P": VALIDATION_DIR / "pilot", "VAL-A": VALIDATION_DIR / "annotation"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefix", default="CAND", help="CAND (evaluation set) or VAL-<letter> (validation)")
    parser.add_argument("--out", type=Path, help="folder (default: by prefix, see above)")
    parser.add_argument("--replace", action="store_true", help="delete the candidate files already in the folder")
    parser.add_argument("--i-know-this-breaks-the-baseline", dest="break_baseline", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    out = args.out or DEFAULT_FOLDERS.get(args.prefix) or VALIDATION_DIR / args.prefix.lower()
    existing = candidate_files(out) if out.exists() else []
    if existing and not args.replace:
        parser.error(f"{out} already holds {len(existing)} candidate files; nothing was written. "
                     "Candidate sets are frozen once used; choose another --out (or --replace deliberately).")
    if existing and out.resolve() == CANDIDATES_DIR.resolve() and not args.break_baseline:
        parser.error("refusing to replace the frozen evaluation candidates in data/candidates/ "
                     "(their hashes are in docs/evaluation_baseline.json)")

    bank = load_bank(ROOT)
    jobs = {j["id"]: j for j in bank.jobs}
    try:
        candidates = make_candidates(bank, n=args.n, seed=args.seed, prefix=args.prefix)
    except ValueError as e:
        parser.error(str(e))
    for old in existing:
        old.unlink()
    save_candidates(candidates, out)

    truths = [truth_for(c, jobs[c["job_id"]]) for c in candidates]
    print(f"Wrote {len(candidates)} candidates ({args.prefix}, seed {args.seed}) to {out}")
    print("per job:", dict(Counter(c["job_id"] for c in candidates)))
    print("fit / not fit:", sum(t["fit"] for t in truths), "/", sum(not t["fit"] for t in truths))
    print("even / uneven:", dict(Counter(c["profile_type"] for c in candidates)))
    print("planted misconceptions:", sum(len(c["misconceptions"]) for c in candidates),
          f"({sum(not c['misconceptions'] for c in candidates)} control candidates with none)")
    print("safety attacks:", [c["id"] for c in candidates if c.get("safety_attack")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
