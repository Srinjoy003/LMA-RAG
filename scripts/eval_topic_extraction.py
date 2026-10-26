r"""Check topic extraction against the answer key in data/jobs.json.

For every job with a description, extract topics and print precision / recall / F1
against `required_topics`, plus the missed and extra topics.

    .venv\Scripts\python.exe scripts\eval_topic_extraction.py

The endpoint and key come from GEMINI_API_KEY (or LLM_BASE_URL / LLM_API_KEY) (environment or the project's .env;
see docs/llm.md). Optional: TOPIC_MODEL=<model id> to use another model. Answers are cached
under .cache/llm, so a re-run makes no calls (LLM_CACHE=0 to force fresh ones).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.bank import load_bank  # noqa: E402
from llm.client import LLMError, get_client  # noqa: E402
from part1.extract_topics import extract_topics, get_model, score_topics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="project root (default: this project)")
    args = parser.parse_args()

    bank = load_bank(args.root)
    try:
        llm = get_client()
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"Model: {get_model()}\n")

    scores = []
    for job in bank.jobs:
        title = f"{job['title']} ({job['id']})"
        if not job.get("description", "").strip():
            print(f"{title}: skipped, empty description\n")
            continue
        try:
            predicted = extract_topics(job["description"], llm=llm, topics=bank.topics)
        except LLMError as e:
            print(f"{title}: FAILED, {e}\n")
            continue
        s = score_topics(predicted, job["required_topics"])
        scores.append(s)
        print(title)
        print(f"  predicted {len(predicted)}, required {len(job['required_topics'])}")
        print(f"  precision {s['precision']:.2f}  recall {s['recall']:.2f}  F1 {s['f1']:.2f}")
        print(f"  missed: {', '.join(s['missed']) or '-'}")
        print(f"  extra:  {', '.join(s['extra']) or '-'}\n")

    if not scores:
        print("No job has a description yet, nothing to evaluate.")
        return 0
    n = len(scores)
    print(f"Average over {n} jobs: precision {sum(s['precision'] for s in scores) / n:.2f}  "
          f"recall {sum(s['recall'] for s in scores) / n:.2f}  F1 {sum(s['f1'] for s in scores) / n:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
