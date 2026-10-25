r"""Live smoke test of the prompted grader and the structured diagnoser (NOT part of pytest).

Picks 6 bank questions (2 per skill: one "explain", one "read-code"). For each it grades
  (a) the model answer           -> should score 2
  (b) a wrong answer taken from one of the question's misconceptions' `example_wrong_answer`
                                 -> should score 0, and the diagnosis should name that misconception
then prints grades, vote agreement and diagnoses, and a summary.

    .venv\Scripts\python.exe scripts\smoke_llm.py [--model gemini-3.5-flash-lite] [--votes 3]
    .venv\Scripts\python.exe scripts\smoke_llm.py --provider groq    # GROQ_API_KEY, its model and effort

Cost: 6 x (2 grades + 1 diagnosis) x votes = 54 calls with the default 3 votes, all cached under
.cache/llm, so an interrupted or repeated run only pays for what is missing. If the provider's daily
quota runs out the script stops and summarises what it has.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.client import DEFAULT_REASONING_EFFORT, LLMError, LLMQuotaError, get_client  # noqa: E402
from llm.providers import PROVIDERS, provider_client, provider_model  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part3 import PromptedGrader, StructuredDiagnoser  # noqa: E402
from shared.interfaces import SKILLS  # noqa: E402


def pick_questions(bank, level: int = 2) -> list[dict]:
    """Per skill: the first 'explain' and the first 'read-code' question at `level` that have a
    misconception with an example wrong answer."""
    chosen = []
    for skill in SKILLS:
        for qtype in ("explain", "read-code"):
            for q in bank.questions_for(skill, level):
                if q["type"] == qtype and wrong_answer_for(bank, q)[0]:
                    chosen.append(q)
                    break
    return chosen


def wrong_answer_for(bank, q: dict) -> tuple[str | None, str | None]:
    """(misconception id, its example wrong answer) for the first listed misconception that has one."""
    for mid in q["misconceptions"]:
        rec = bank.get_misconception(mid)
        if rec and rec.get("example_wrong_answer"):
            return mid, rec["example_wrong_answer"]
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=sorted(PROVIDERS),
                    help="use this provider's endpoint, key, model and effort (default: LLM_BASE_URL / LLM_API_KEY)")
    ap.add_argument("--model", default=None, help="default: the provider's model, else gemini-3.5-flash-lite")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--effort", default=None,
                    help=f"reasoning_effort (default: the provider's, else {DEFAULT_REASONING_EFFORT})")
    ap.add_argument("--level", type=int, default=2)
    args = ap.parse_args()
    if args.provider:     # the provider's own effort, even when it has none (Ollama)
        args.model = args.model or provider_model(args.provider)
        args.effort = args.effort or PROVIDERS[args.provider].reasoning_effort
    else:
        args.effort = args.effort or DEFAULT_REASONING_EFFORT
    args.model = args.model or "gemini-3.5-flash-lite"

    bank = load_bank(ROOT)
    records = {m["id"]: m for m in bank.misconceptions}
    try:
        llm = provider_client(args.provider) if args.provider else get_client()
    except RuntimeError as e:
        sys.exit(str(e))
    grader = PromptedGrader(llm, args.model, votes=args.votes, reasoning_effort=args.effort, misconceptions=records)
    diagnoser = StructuredDiagnoser(llm, args.model, records, votes=args.votes, reasoning_effort=args.effort)
    print(f"model {args.model}  votes {args.votes}  reasoning_effort {args.effort}\n")

    correct_ok = wrong_ok = cause_ok = id_ok = n_correct = n_wrong = 0
    agreements: list[float] = []
    try:
        for q in pick_questions(bank, args.level):
            mid, wrong = wrong_answer_for(bank, q)
            print(f"== {q['id']} ({q['type']}, {', '.join(q['topics'])})  planted: {mid}")

            g = grader.grade(q, q["model_answer"])
            n_correct += 1
            correct_ok += g["score"] == 2
            agreements.append(g["agreement"])
            print(f"  model answer : score {g['score']} votes {g['votes']} agreement {g['agreement']:.2f}")

            g = grader.grade(q, wrong)
            n_wrong += 1
            wrong_ok += g["score"] == 0
            agreements.append(g["agreement"])
            print(f"  wrong answer : score {g['score']} votes {g['votes']} agreement {g['agreement']:.2f}")
            print(f"    answer   : {wrong}")
            print(f"    rationale: {g['rationale']}")

            d = diagnoser.diagnose(q, wrong)
            cause_ok += d["cause"] == "misconception"
            id_ok += d["cause"] == "misconception" and d["misconception_id"] == mid
            print(f"  diagnosis    : {d['cause']} / {d['misconception_id']}  vote_share {d['vote_share']:.2f}  "
                  f"confidence {d['confidence_expressed']}")
            print(f"    votes    : {[(v['cause'], v['misconception_id']) for v in d['votes']]}")
            print(f"    follow_up: {d['follow_up']}\n")
    except LLMQuotaError as e:
        print(f"\nSTOPPED, quota: {e}\n")
    except LLMError as e:
        print(f"\nSTOPPED, LLM error: {e}\n")

    print("== Summary")
    print(f"  model answers scored 2     : {correct_ok}/{n_correct}")
    print(f"  wrong answers scored 0     : {wrong_ok}/{n_wrong}")
    print(f"  diagnosis cause=misconception : {cause_ok}/{n_wrong}")
    print(f"  diagnosis named the planted id: {id_ok}/{n_wrong}")
    if agreements:
        print(f"  mean grading agreement     : {sum(agreements) / len(agreements):.2f}")
    print(f"  network calls {llm.stats['calls']} (retries {llm.stats['retries']}), cache hits {llm.stats['cache_hits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
