r"""Misconception recall with the stubs: A2 with `probe_missed_topics` off and on.

    .venv\Scripts\python.exe scripts\misconception_recall.py
    .venv\Scripts\python.exe scripts\misconception_recall.py --candidates 10 --seeds 5

Makes candidates from examples/candidate.example.json (random topic levels, 1-2 misconceptions
each, planted only on the job's required topics), runs A2 for each, and counts how many planted
misconceptions were probed (listed on an asked question) and how many were confirmed.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.run import run_interview  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part1.index import DEFAULT_INDEX_DIR, QuestionIndex  # noqa: E402
from shared.interfaces import SYSTEMS, Components  # noqa: E402
from shared.stubs import RuleBasedRespondent, TagDiagnoser, TagGrader  # noqa: E402


def make_candidates(bank, example: dict, n: int = 10, seed: int = 123) -> list[dict]:
    """n candidates over the jobs in turn, each with 1-2 misconceptions on the job's required topics."""
    rng = random.Random(seed)
    jobs = [j["id"] for j in bank.jobs]
    out = []
    for i in range(n):
        job = bank.get_job(jobs[i % len(jobs)])
        levels = {t: rng.choice([0, 1, 2, 2, 3, 3]) for t in example["topic_levels"]}
        pool = [m for m in bank.misconceptions if m["topic"] in job["required_topics"]]
        planted = [m["id"] for m in rng.sample(pool, rng.choice([1, 2]))]
        out.append({**example, "id": f"CAND-{i + 1:03d}", "job_id": job["id"], "topic_levels": levels,
                    "misconceptions": planted, "misread_rate": 0.0, "slip_rate": 0.05, "seed": i + 1})
    return out


def run_all(bank, source, candidates: list[dict], probe: bool, seeds=(0,), system: str = "A2") -> list[dict]:
    rows = []
    for candidate in candidates:
        for seed in seeds:
            config = SYSTEMS[system].with_(seed=seed, probe_missed_topics=probe)
            components = Components(question_source=source, grader=TagGrader(votes=config.grader_votes),
                                    diagnoser=TagDiagnoser(), misconceptions=bank.misconception_by_id)
            result = run_interview(bank.get_job(candidate["job_id"]), RuleBasedRespondent(candidate), config,
                                   components, candidate_id=candidate["id"], rng=random.Random(seed))
            report = result["report"]
            confirmed = {m["id"] for m in report["misconceptions"] if m["confirmed"]}
            planted = set(candidate["misconceptions"])
            rows.append({"candidate": candidate["id"], "seed": seed, "planted": len(planted),
                         "probed": len(planted & set(report["probed_misconceptions"])),
                         "confirmed": len(planted & confirmed), "false_positives": len(confirmed - planted),
                         "main_questions": len(result["turns"]), "exchanges": result["exchanges"]})
    return rows


def summarise(rows: list[dict]) -> dict:
    return {"runs": len(rows), "planted": sum(r["planted"] for r in rows),
            "probed": sum(r["probed"] for r in rows), "confirmed": sum(r["confirmed"] for r in rows),
            "false_positives": sum(r["false_positives"] for r in rows),
            "mean_main_questions": round(sum(r["main_questions"] for r in rows) / len(rows), 2),
            "mean_exchanges": round(sum(r["exchanges"] for r in rows) / len(rows), 2)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=1, help="interview seeds per candidate (0..n-1)")
    parser.add_argument("--system", default="A2", choices=sorted(SYSTEMS))
    args = parser.parse_args(argv)

    bank = load_bank()
    source = QuestionIndex.load(DEFAULT_INDEX_DIR, bank=bank)
    example = json.loads((ROOT / "examples" / "candidate.example.json").read_text(encoding="utf-8"))
    candidates = make_candidates(bank, example, args.candidates)
    summary = {}
    for probe in (False, True):
        rows = run_all(bank, source, candidates, probe, tuple(range(args.seeds)), args.system)
        summary[probe] = summarise(rows)
        s = summary[probe]
        print(f"{args.system} probing {'on ' if probe else 'off'}: planted {s['planted']}  "
              f"probed {s['probed']}  confirmed {s['confirmed']}  false positives {s['false_positives']}  "
              f"main questions {s['mean_main_questions']}  exchanges {s['mean_exchanges']}")
    return summary


if __name__ == "__main__":
    main()
