"""Growing the bank from the drafting pool (part1.pool). Small hand-made data, no model."""

import numpy as np

from part1.pool import choose_promotions, eligible, renumber


def question(qid, kind="explain", misconceptions=("py.a.x",), text=None):
    skill, level, _ = qid.split("-")
    return {"id": qid, "skill": skill.lower(), "level": int(level[1]), "type": kind,
            "topics": ["py-a"], "question": text or qid, "misconceptions": list(misconceptions)}


def review(score, selected=False):
    return {"score": score, "selected": selected, "reason": "fine"}


def test_eligible_keeps_good_kept_back_questions_not_yet_promoted():
    pool = [question(f"PY-L1-10{i}") for i in range(1, 5)] + [question("SQL-L3-103")]
    reviews = {"PY-L1-101": review(5), "PY-L1-102": review(3), "PY-L1-103": review(5, selected=True),
               "PY-L1-104": review(4), "SQL-L3-103": review(5)}           # SQL-L3-103: level misfit
    record = [{"blind_id": "PY-L1-104"}]                                    # already promoted
    assert [q["id"] for q in eligible(pool, reviews, record)] == ["PY-L1-101"]


def test_thin_misconceptions_come_first_then_score():
    bank = [question("PY-L1-001", misconceptions=["py.a.common"]) for _ in range(3)]
    candidates = [question("PY-L1-101", misconceptions=["py.a.common"]),
                  question("PY-L1-102", misconceptions=["py.a.rare"]),
                  question("PY-L1-103", misconceptions=["py.a.rare"])]
    reviews = {"PY-L1-101": review(5), "PY-L1-102": review(4), "PY-L1-103": review(5)}
    chosen = choose_promotions(bank, candidates, reviews, per_cell={"explain": 2})
    assert [q["id"] for q in chosen] == ["PY-L1-103", "PY-L1-102"]


def test_types_are_filled_separately():
    candidates = [question("PY-L2-101", "read-code"), question("PY-L2-102"), question("PY-L2-103")]
    reviews = {q["id"]: review(4) for q in candidates}
    chosen = choose_promotions([], candidates, reviews, per_cell={"explain": 1, "read-code": 1})
    assert sorted(q["type"] for q in chosen) == ["explain", "read-code"]


def test_near_duplicates_are_skipped():
    bank = [question("PY-L1-001", text="same words")]
    candidates = [question("PY-L1-101", text="same words"), question("PY-L1-102", text="other")]
    reviews = {"PY-L1-101": review(5), "PY-L1-102": review(4)}
    embed = lambda texts: np.array([[1.0, 0.0] if "same words" in t else [0.0, 1.0] for t in texts])
    chosen = choose_promotions(bank, candidates, reviews, embed_fn=embed, per_cell={"explain": 1})
    assert [q["id"] for q in chosen] == ["PY-L1-102"]


def test_renumber_continues_each_cell():
    bank = [question("PY-L1-009"), question("PY-L1-010"), question("PY-L2-010")]
    promoted = renumber([question("PY-L1-117"), question("PY-L1-103"), question("PY-L2-111")], bank)
    assert [(blind, q["id"]) for blind, q in promoted] == [
        ("PY-L1-117", "PY-L1-011"), ("PY-L1-103", "PY-L1-012"), ("PY-L2-111", "PY-L2-011")]
