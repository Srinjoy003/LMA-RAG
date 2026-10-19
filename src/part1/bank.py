"""Load the question bank, misconceptions, topics and jobs from disk.

Missing files are treated as empty, so the loader never crashes on a bank
that is still being written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SKILLS = ("py", "sql", "ds")
LEVELS = (1, 2, 3)

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path, default):
    """Read a JSON file; return `default` if the file does not exist."""
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def question_text(q: dict, topic_name=lambda t: t) -> str:
    """The text that represents a question in the search index:
    question text, code, topic names and key points."""
    parts = [q.get("question", "")]
    if q.get("code"):
        parts.append(q["code"])
    parts.append("Topics: " + ", ".join(topic_name(t) for t in q.get("topics", [])))
    parts.append("Key points: " + "; ".join(q.get("key_points", [])))
    return "\n".join(parts)


@dataclass
class Bank:
    root: Path
    topics: dict = field(default_factory=dict)  # skill -> list of topic dicts
    jobs: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    misconceptions: list = field(default_factory=list)
    # files that existed but could not be parsed: path -> error message
    load_errors: dict = field(default_factory=dict)

    # ---- lookups -------------------------------------------------------
    def __post_init__(self):
        self._index()

    def _index(self):
        self.question_by_id = {}
        for q in self.questions:
            self.question_by_id.setdefault(q.get("id"), q)
        self.misconception_by_id = {}
        for m in self.misconceptions:
            self.misconception_by_id.setdefault(m.get("id"), m)

    def topic_ids(self, skill: str | None = None) -> set[str]:
        skills = [skill] if skill else list(self.topics)
        return {t["id"] for s in skills for t in self.topics.get(s, [])}

    def topic_skill(self, topic_id: str) -> str | None:
        for skill, items in self.topics.items():
            if any(t["id"] == topic_id for t in items):
                return skill
        return None

    def topic_name(self, topic_id: str) -> str:
        for items in self.topics.values():
            for t in items:
                if t["id"] == topic_id:
                    return t["name"]
        return topic_id

    def get_question(self, question_id: str) -> dict | None:
        return self.question_by_id.get(question_id)

    def get_misconception(self, misconception_id: str) -> dict | None:
        return self.misconception_by_id.get(misconception_id)

    def get_job(self, job_id: str) -> dict | None:
        return next((j for j in self.jobs if j.get("id") == job_id), None)

    def questions_for(self, skill: str | None = None, level: int | None = None) -> list[dict]:
        return [
            q for q in self.questions
            if (skill is None or q.get("skill") == skill)
            and (level is None or q.get("level") == level)
        ]

    def questions_by_topic(self, topic_id: str) -> list[dict]:
        return [q for q in self.questions if topic_id in q.get("topics", [])]

    def misconceptions_for(self, skill: str | None = None, topic: str | None = None) -> list[dict]:
        return [
            m for m in self.misconceptions
            if (skill is None or m.get("skill") == skill)
            and (topic is None or m.get("topic") == topic)
        ]

    def required_topics(self) -> set[str]:
        """Union of required topics over all jobs."""
        return {t for j in self.jobs for t in j.get("required_topics", [])}

    def content_hash(self) -> str:
        """Hash of the text that gets embedded, so a stale search index can be detected."""
        texts = sorted((str(q.get("id")), question_text(q, self.topic_name)) for q in self.questions)
        payload = json.dumps(texts, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_bank(root: str | Path | None = None, skills=SKILLS) -> Bank:
    """Load everything under `root` (default: the project root)."""
    root = Path(root) if root else ROOT
    data = root / "data"
    errors: dict = {}

    def load_list(path: Path) -> list:
        try:
            value = _read_json(path, [])
        except (json.JSONDecodeError, OSError) as e:
            errors[str(path)] = str(e)
            return []
        if not isinstance(value, list):
            errors[str(path)] = "expected a JSON array"
            return []
        return value

    def load_obj(path: Path, default):
        try:
            return _read_json(path, default)
        except (json.JSONDecodeError, OSError) as e:
            errors[str(path)] = str(e)
            return default

    topics = load_obj(data / "topics.json", {})
    jobs = load_obj(data / "jobs.json", [])
    questions, misconceptions = [], []
    for skill in skills:
        questions += load_list(data / "questions" / f"{skill}.json")
        misconceptions += load_list(data / "misconceptions" / f"{skill}.json")
    return Bank(root=root, topics=topics, jobs=jobs, questions=questions,
                misconceptions=misconceptions, load_errors=errors)
