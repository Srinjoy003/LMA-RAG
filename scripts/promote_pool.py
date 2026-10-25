r"""Grow the bank with the best kept-back questions from data/pool/ (see src/part1/pool.py).

    .venv/bin/python scripts/promote_pool.py            # dry run: show what would be added
    .venv/bin/python scripts/promote_pool.py --write    # add them to data/questions/ and the record

Adds 3 explain + 2 read-code questions per skill-level cell (90 -> 135), skipping near-duplicates
(loads the embedding model). Promoted questions stay `draft` until a teammate checks them.
Afterwards run scripts/validate_bank.py and scripts/build_index.py.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import load_bank  # noqa: E402
from part1.index import default_embedder  # noqa: E402
from part1.pool import (THIN, choose_promotions, eligible, load_pool, record_entries,  # noqa: E402
                        renumber, write_promotions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the changes (default: dry run)")
    args = parser.parse_args()

    bank = load_bank(ROOT)
    pool, reviews, record = load_pool(ROOT)
    candidates = eligible(pool, reviews, record)
    chosen = choose_promotions(bank.questions, candidates, reviews, embed_fn=default_embedder())
    promoted = renumber(chosen, bank.questions)

    for blind_id, q in promoted:
        print(f"{q['id']}  <- {blind_id}  {q['type']:<9}  score {reviews[blind_id]['score']}  "
              f"{', '.join(q['misconceptions'])}")
    before = Counter(m for q in bank.questions for m in q["misconceptions"])
    after = before + Counter(m for q in chosen for m in q["misconceptions"])
    ids = [m["id"] for m in bank.misconceptions]
    print(f"\n{len(candidates)} eligible, {len(promoted)} chosen: bank {len(bank.questions)} -> "
          f"{len(bank.questions) + len(promoted)} questions")
    print(f"misconceptions on {THIN}+ questions: {sum(before[m] >= THIN for m in ids)} -> "
          f"{sum(after[m] >= THIN for m in ids)} of {len(ids)}")

    if not args.write:
        print("\nDry run: nothing written. Add --write to apply.")
        return 0
    write_promotions(promoted, record_entries(promoted, reviews, ROOT), ROOT)
    print("\nWritten. Next: scripts/validate_bank.py, then scripts/build_index.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
