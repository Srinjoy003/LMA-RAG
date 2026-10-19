r"""Embed every question and save the search index to index/.

    .venv\Scripts\python.exe scripts\build_index.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import load_bank  # noqa: E402
from part1.index import MODEL_NAME, QuestionIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project root (default: this project)")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <root>/index)")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    bank = load_bank(args.root)
    if not bank.questions:
        print("No questions found under data/questions/ - nothing to index.")
        return 1
    out = args.out or args.root / "index"
    print(f"Embedding {len(bank.questions)} questions with {args.model} ...")
    index = QuestionIndex.build(bank, model_name=args.model)
    index.save(out)
    print(f"Saved {out / 'embeddings.npy'} and {out / 'meta.json'} "
          f"(shape {index.embeddings.shape}, hash {index.bank_hash[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
