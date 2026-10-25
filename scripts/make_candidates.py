r"""Generate the simulated candidates' hidden keys into data/candidates/ (see src/part4/candidates.py).

    .venv/bin/python scripts/make_candidates.py            # 60 candidates, seed 2026
    .venv/bin/python scripts/make_candidates.py --n 12     # a smaller set

Deterministic: the same --n and --seed always give the same candidates. Existing files are replaced.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import load_bank  # noqa: E402
from part4.candidates import CANDIDATES_DIR, make_candidates, save_candidates, truth_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    bank = load_bank(ROOT)
    jobs = {j["id"]: j for j in bank.jobs}
    candidates = make_candidates(bank, n=args.n, seed=args.seed)
    for old in CANDIDATES_DIR.glob("CAND-*.json"):
        old.unlink()
    save_candidates(candidates)

    truths = [truth_for(c, jobs[c["job_id"]]) for c in candidates]
    print(f"Wrote {len(candidates)} candidates to {CANDIDATES_DIR}")
    print("per job:", dict(Counter(c["job_id"] for c in candidates)))
    print("fit / not fit:", sum(t["fit"] for t in truths), "/", sum(not t["fit"] for t in truths))
    print("even / uneven:", dict(Counter(c["profile_type"] for c in candidates)))
    print("planted misconceptions:", sum(len(c["misconceptions"]) for c in candidates),
          f"({sum(not c['misconceptions'] for c in candidates)} control candidates with none)")
    print("safety attacks:", [c["id"] for c in candidates if c.get("safety_attack")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
