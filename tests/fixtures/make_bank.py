"""Builds a small, fully valid bank on disk for the tests.

Everything is generated: 5 fake topics per skill, 30 questions per skill (10 per
level, 6 explain + 4 read-code), 5 misconceptions per skill, and one job that
requires one topic per skill. Tests load it, change one thing, and check that
exactly the intended rule fires.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILLS = {"py": "PY", "sql": "SQL", "ds": "DS"}
SUFFIXES = ["alpha", "beta", "gamma", "delta", "epsilon"]


def topic_ids(skill):
    return [f"{skill}-{s}" for s in SUFFIXES]


def misconception_ids(skill):
    return [f"{skill}.{s}.wrong-idea" for s in SUFFIXES]


def make_question(skill, level, i):
    """Question number i (1..10) of one skill-level cell."""
    topics = topic_ids(skill)
    miscs = misconception_ids(skill)
    k = (i + level) % 5
    q = {
        "id": f"{SKILLS[skill]}-L{level}-{i:03d}",
        "skill": skill,
        "level": level,
        "type": "explain" if i <= 6 else "read-code",
        "topics": [topics[k], topics[(k + 1) % 5]],
        # many words that are unique to this question, so the bag-of-words test embedder
        # sees different questions as dissimilar
        "question": "Explain " + " ".join(f"u{skill}{level}x{i}x{n}" for n in range(15)),
        "model_answer": "A model answer.",
        "key_points": ["first point", "second point"],
        "rubric": {"0": "wrong", "1": "partly right", "2": "right"},
        "misconceptions": [miscs[k]],
        "source": "generated",
        "status": "draft",
    }
    if q["type"] == "read-code":
        if skill == "sql":
            q["code"] = "SELECT id, name FROM people ORDER BY id;"
            q["tables"] = [{"name": "people", "columns": ["id", "name"],
                            "rows": [[1, "Asha"], [2, "Ben"], [3, None]]}]
            q["expected_output"] = [[1, "Asha"], [2, "Ben"], [3, None]]
        else:
            q["code"] = f"print({i} * 2)"
            q["expected_output"] = f"{i * 2}\n"
    return q


def write_bank(root: Path, skills=("py", "sql", "ds")) -> Path:
    root = Path(root)
    (root / "data" / "questions").mkdir(parents=True)
    (root / "data" / "misconceptions").mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "schemas", root / "schemas")

    topics = {s: [{"id": t, "name": t.split("-")[1].title() + " topic", "typical_levels": [1, 2, 3]}
                  for t in topic_ids(s)] for s in SKILLS}
    jobs = [{
        "id": "job-1", "title": "Test Job", "description": "Write queries and scripts all day.",
        "min_levels": {"py": 1, "sql": 1, "ds": 1},
        "required_topics": [topic_ids(s)[0] for s in SKILLS],
    }]
    (root / "data" / "topics.json").write_text(json.dumps(topics), encoding="utf-8")
    (root / "data" / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")

    for skill in skills:
        questions = [make_question(skill, level, i) for level in (1, 2, 3) for i in range(1, 11)]
        miscs = [{"id": mid, "skill": skill, "topic": f"{skill}-{mid.split('.')[1]}",
                  "description": "A wrong idea.", "example_wrong_answer": "Wrong.", "source": "generated"}
                 for mid in misconception_ids(skill)]
        (root / "data" / "questions" / f"{skill}.json").write_text(json.dumps(questions), encoding="utf-8")
        (root / "data" / "misconceptions" / f"{skill}.json").write_text(json.dumps(miscs), encoding="utf-8")
    return root
