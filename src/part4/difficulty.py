"""Reproducible blind difficulty packets; no LLM calls and no answer-key output."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import subprocess
from pathlib import Path

SKILLS = ("py", "sql", "ds")
SEED = 1401
PER_SKILL = 7


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fenced(text: str, language: str = "") -> str:
    fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", text)), default=3))
    return f"{fence}{language}\n{text}\n{fence}"


def prepare_difficulty(root: Path, output: Path, *, seed: int = SEED,
                       per_skill: int = PER_SKILL) -> dict:
    """Create a fresh packet, independent sheets and private provenance/key files.

    Sampling uses ID-sorted inputs, one RNG across py/sql/ds and a final shuffle.
    Returns only public paths/counts; never returns the question-ID mapping.
    Existing output directories are refused, including completed rating sheets.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if isinstance(per_skill, bool) or not isinstance(per_skill, int) or per_skill < 1:
        raise ValueError("per_skill must be a positive integer")
    rng = random.Random(seed)
    selected, hashes, seen = [], {}, set()
    for skill in SKILLS:
        path = root / "data" / "questions" / f"{skill}.json"
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = _sha(raw)
        questions = json.loads(raw)
        if not isinstance(questions, list) or len(questions) < per_skill:
            raise ValueError(f"{skill}: insufficient questions for the requested sample")
        for q in questions:
            if q["id"] in seen:
                raise ValueError("duplicate question ID in bank")
            seen.add(q["id"])
            if q["skill"] != skill or q["status"] != "checked" or q["level"] not in (1, 2, 3):
                raise ValueError(f"{skill}: bank must contain checked questions of the correct skill/level")
        selected.extend(rng.sample(sorted(questions, key=lambda q: q["id"]), per_skill))
    rng.shuffle(selected)

    packet = ["# Blind difficulty check", "",
              "Assign each item level 1, 2 or 3 using the project's written level definitions.",
              "Work independently. Record ratings in your own CSV; do not open private files or look up bank items.", ""]
    key = []
    for i, q in enumerate(selected, 1):
        item = f"D{i:02d}"
        packet.extend([f"## {item}", "", q["question"], ""])
        if q.get("code"):
            packet.extend([_fenced(q["code"], "sql" if q["skill"] == "sql" else "python"), ""])
        for table in q.get("tables") or []:
            rows = [" | ".join(map(str, table["columns"]))]
            rows.extend(" | ".join("NULL" if v is None else str(v) for v in row) for row in table["rows"])
            packet.extend([f"Table: {table['name']}", "", _fenced("\n".join(rows)), ""])
        packet.extend(["### Reference answer", "", q["model_answer"], "",
                       "### Key points", ""])
        packet.extend(f"- {point}" for point in q["key_points"])
        packet.append("")
        key.append({"item": item, "question_id": q["id"], "skill": q["skill"], "assigned_level": q["level"]})

    public = {"packet.md": "\n".join(packet)}
    for rater in ("souradeep", "nilkanta"):
        sheet = io.StringIO(newline="")
        writer = csv.writer(sheet, lineterminator="\n")
        writer.writerow(["item", "rater", "level", "note"])
        writer.writerows([row["item"], rater, "", ""] for row in key)
        public[f"ratings/{rater}.csv"] = sheet.getvalue()
    public["instructions.md"] = """# Difficulty check instructions

The preparer distributes packet.md and only the recipient's ratings CSV.
Both raters use the same packet and the level definitions in docs/part1_spec.md, section 5.
Fill level (1, 2 or 3) and optional note; do not change item/rater columns.
Do not inspect the private directory, bank files, or the other person's ratings.
Return your completed CSV to the preparer; keep independent submissions unchanged.
The preparer hashes and freezes both submissions before opening the key for scoring.
Agreement scoring and adjudication are a later task; this generator assigns no ratings.
The packet deliberately includes reference answers and key points, as specified by the protocol.
Opaque item codes conceal assigned metadata, not conceptual clues in the question itself.
Use Kushal or Srinjoy as preparer where possible, since Souradeep and Nilkanta are raters.
Do not rerun with another seed to select more convenient items.
"""
    # Fail rather than accidentally expose an original ID embedded in bank text.
    if any(qid in public["packet.md"] for qid in seen):
        raise ValueError("packet text contains a bank question ID; inspect the source before distribution")
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    key_text = json.dumps(key, indent=2) + "\n"
    manifest = {"format_version": 1, "task": "blind_difficulty", "seed": seed,
                "per_skill": per_skill, "items": len(key), "source_commit": commit,
                "sampling": "ID-sort each skill; random.Random(seed).sample in py/sql/ds order; shuffle once",
                "source_files_sha256": hashes,
                "generator_sha256": _sha(Path(__file__).read_bytes()),
                "key_sha256": _sha(key_text.encode()),
                "public_files_sha256": {name: _sha(text.encode()) for name, text in public.items()},
                "note": "Commit identifies HEAD; data/generator hashes identify actual working-tree inputs."}
    # Exclusive directory creation prevents reruns from replacing any existing packet or labels.
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    (output / "private").mkdir(mode=0o700)
    for name, text in public.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    for name, text in {"key.json": key_text, "manifest.json": json.dumps(manifest, indent=2) + "\n"}.items():
        path = output / "private" / name
        path.write_bytes(text.encode("utf-8"))
        path.chmod(0o600)
    return {"items": len(key), "per_skill": per_skill, "output": str(output),
            "packet": str(output / "packet.md")}
