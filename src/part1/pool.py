"""Grow the bank from the drafting pool (revision plan, Change 1: 90 -> 120-150 questions).

The bank was chosen from 180 drafted questions in data/pool/: 90 selected, 90 kept back,
all reviewed blind in data/pool/selection.json. `choose_promotions` picks the best kept-back
questions for every skill-level cell, preferring those that list the misconceptions the bank
covers least, so each misconception can be probed more often.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .bank import LEVELS, ROOT, SKILLS

MIN_SCORE = 4                                     # review score out of 5
LEVEL_MISFITS = frozenset({"SQL-L3-103", "SQL-L3-120"})   # the review calls these L2 content
PER_CELL = {"explain": 3, "read-code": 2}         # added per cell: keeps the bank's 3:2 split
THIN = 3                                          # a misconception on fewer questions is thin
DUPLICATE_AT = 0.9                                # cosine similarity, as in the validator


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_pool(root: Path = ROOT) -> tuple[list[dict], dict, list[dict]]:
    """(blind pool questions, blind reviews by id, selection record)."""
    pool_dir = Path(root) / "data" / "pool"
    pool = [q for skill in SKILLS for q in _read(pool_dir / "blind" / f"{skill}.json")]
    return pool, _read(pool_dir / "selection.json")["reviews"], _read(pool_dir / "selection_record.json")


def eligible(pool: list[dict], reviews: dict, record: list[dict]) -> list[dict]:
    """Kept-back questions with a good review that are not in the bank yet."""
    in_bank = {r["blind_id"] for r in record}
    return [q for q in pool
            if q["id"] not in in_bank and q["id"] not in LEVEL_MISFITS
            and not reviews[q["id"]]["selected"] and reviews[q["id"]]["score"] >= MIN_SCORE]


def choose_promotions(bank_questions: list[dict], candidates: list[dict], reviews: dict,
                      embed_fn=None, per_cell: dict = PER_CELL) -> list[dict]:
    """For each skill, level and type, up to `per_cell[type]` candidates, best first:
    most thin misconceptions covered, then higher review score, then id. A candidate that is
    a near-duplicate of a bank question or an earlier pick is skipped (needs `embed_fn`)."""
    coverage = Counter(m for q in bank_questions for m in q["misconceptions"])
    vectors = _unit_vectors(bank_questions + candidates, embed_fn)
    kept = [q["id"] for q in bank_questions]
    chosen = []
    for skill in SKILLS:
        for level in LEVELS:
            for kind, wanted in per_cell.items():
                pool = [q for q in candidates if (q["skill"], q["level"], q["type"]) == (skill, level, kind)]
                while wanted and pool:
                    pool.sort(key=lambda q: (-sum(coverage[m] < THIN for m in q["misconceptions"]),
                                             -reviews[q["id"]]["score"], q["id"]))
                    pick = pool.pop(0)
                    if _is_duplicate(pick["id"], kept, vectors):
                        continue
                    chosen.append(pick)
                    kept.append(pick["id"])
                    coverage.update(pick["misconceptions"])
                    wanted -= 1
    return chosen


def renumber(chosen: list[dict], bank_questions: list[dict]) -> list[tuple[str, dict]]:
    """[(blind id, question with its new bank id)], numbered on from each cell's last id."""
    last = Counter()
    for q in bank_questions:
        prefix, number = q["id"].rsplit("-", 1)
        last[prefix] = max(last[prefix], int(number))
    out = []
    for q in chosen:
        prefix = q["id"].rsplit("-", 1)[0]
        last[prefix] += 1
        out.append((q["id"], {**q, "id": f"{prefix}-{last[prefix]:03d}"}))
    return out


def record_entries(promoted: list[tuple[str, dict]], reviews: dict, root: Path = ROOT) -> list[dict]:
    """selection_record.json entries for the promoted questions, traced back to their batch."""
    pool_dir = Path(root) / "data" / "pool"
    origin = {q["question"]: (batch, q["id"]) for batch in ("batch_a", "batch_b")
              for skill in SKILLS for q in _read(pool_dir / batch / f"{skill}.json")}
    entries = []
    for blind_id, q in promoted:
        batch, original_id = origin[q["question"]]
        entries.append({"final_id": q["id"], "blind_id": blind_id, "batch": batch, "original_id": original_id,
                        "score": reviews[blind_id]["score"], "reason": reviews[blind_id]["reason"],
                        "promoted": True})
    return entries


def write_promotions(promoted: list[tuple[str, dict]], entries: list[dict], root: Path = ROOT) -> None:
    """Add the questions to data/questions/<skill>.json (sorted by id) and the selection record."""
    data = Path(root) / "data"
    for skill in SKILLS:
        new = [q for _, q in promoted if q["skill"] == skill]
        if new:
            path = data / "questions" / f"{skill}.json"
            _write(path, sorted(_read(path) + new, key=lambda q: q["id"]))
    record_path = data / "pool" / "selection_record.json"
    _write(record_path, _read(record_path) + entries)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _unit_vectors(questions: list[dict], embed_fn) -> dict | None:
    if embed_fn is None:
        return None
    from .bank import question_text
    matrix = np.asarray(embed_fn([question_text(q) for q in questions]), dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return dict(zip((q["id"] for q in questions), matrix))


def _is_duplicate(question_id: str, kept: list[str], vectors: dict | None) -> bool:
    if vectors is None or not kept:
        return False
    return max(float(vectors[question_id] @ vectors[k]) for k in kept) > DUPLICATE_AT
