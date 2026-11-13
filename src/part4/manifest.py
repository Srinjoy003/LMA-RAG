"""Run manifests: what produced a results folder, and refusal to mix anything else into it.

Every run folder `results/<run>/` gets a `manifest.json` before its first interview:

    fingerprint   must be identical for every interview in the run, on every machine:
                  hashes of the engine source (src/**/*.py), the bank inputs, the prompt and
                  evidence sources, and the grader/simulator setup
    conditions    each system's full `config.to_dict()` as run (added the first time it runs)
    candidates    the hash of each candidate file interviewed
    sessions      who ran what and when (operator, share, candidate ids, git commit), append-only

Opening a run whose fingerprint, a condition's configuration or a candidate's file differs
raises `RunManifestError`: start a new run name instead (protocol section 4). A folder that
already holds results but no manifest (for example the historical `midsub`) is refused, so
an old folder is never silently extended with new interviews.

Hashes are of text with CRLF turned into LF, so a Windows checkout (core.autocrlf) and a
macOS checkout of the same commit give the same fingerprint. These match the hashes in
docs/evaluation_baseline.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
FAILURES = "failures.jsonl"
BANK_FILES = ("data/questions/py.json", "data/questions/sql.json", "data/questions/ds.json",
              "data/misconceptions/py.json", "data/misconceptions/sql.json", "data/misconceptions/ds.json",
              "data/topics.json", "data/jobs.json")
# Shown separately for readability; all of them are also inside the code hash.
PROMPT_SOURCES = ("src/part3/prompts.py", "src/part3/diagnoser.py", "src/part3/grader.py",
                  "src/part4/simulator.py", "src/shared/evidence.py", "src/shared/stubs.py",
                  "src/part1/extract_topics.py", "src/engine/agent.py", "src/engine/followups.py")


class RunManifestError(RuntimeError):
    """The run folder was made by something else; `problems` says what differs."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def text_sha256(data: bytes) -> str:
    """SHA-256 of text with CRLF line endings normalised to LF."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: Path) -> str:
    return text_sha256(Path(path).read_bytes())


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def code_hashes(code_root: Path) -> dict[str, str]:
    """Every engine source file under src/, by path relative to the repository."""
    src = Path(code_root) / "src"
    return {p.relative_to(code_root).as_posix(): file_sha256(p)
            for p in sorted(src.rglob("*.py")) if "__pycache__" not in p.parts}


def bank_hashes(bank_root: Path) -> dict[str, str]:
    root = Path(bank_root)
    return {name: file_sha256(root / name) for name in BANK_FILES if (root / name).exists()}


def git_state(code_root: Path) -> dict:
    """Commit and whether src/ or data/ have uncommitted changes (information only)."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(code_root), *args], text=True,
                                       stderr=subprocess.DEVNULL, timeout=10).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src", "data"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def fingerprint(*, code_root: Path, bank_root: Path, setup: dict) -> dict:
    code = code_hashes(code_root)
    return {"code_sha256": canonical_sha256(code), "code_files": code,
            "bank_files_sha256": bank_hashes(bank_root),
            "prompt_sources_sha256": {name: code[name] for name in PROMPT_SOURCES if name in code},
            "setup": dict(setup)}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _diff(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Paths of the keys whose values differ, for a readable refusal."""
    out = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff(a, b, f"{prefix}{key}.")
        elif a != b:
            out.append(f"{prefix}{key}")
    return out


# --------------------------------------------------------------------------
# The manifest of one run folder
# --------------------------------------------------------------------------

class RunManifest:
    def __init__(self, run_dir: Path, data: dict):
        self.run_dir, self.data = Path(run_dir), data

    @property
    def path(self) -> Path:
        return self.run_dir / MANIFEST

    @property
    def fingerprint_id(self) -> str:
        return self.data["fingerprint_id"]

    @classmethod
    def load(cls, run_dir: Path) -> "RunManifest | None":
        path = Path(run_dir) / MANIFEST
        if not path.exists():
            return None
        return cls(run_dir, json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def open(cls, run_dir: Path, *, code_root: Path, bank_root: Path, setup: dict) -> "RunManifest":
        """The manifest of `run_dir`, created if the folder is new. Refuses a different setup,
        and a folder that holds results without a manifest."""
        run_dir = Path(run_dir)
        current = fingerprint(code_root=code_root, bank_root=bank_root, setup=setup)
        existing = cls.load(run_dir)
        if existing is None:
            if any(run_dir.glob("*/*/*.json")):
                raise RunManifestError([f"{run_dir} already holds results but no {MANIFEST}: it was made before "
                                        "run versioning (or by hand). Use a new --run name."])
            return cls(run_dir, {"format_version": FORMAT_VERSION, "run": run_dir.name, "created_at": now_iso(),
                                 "fingerprint_id": canonical_sha256(current), "fingerprint": current,
                                 "conditions": {}, "candidates": {}, "sessions": []})
        differs = _diff(existing.data["fingerprint"], current)
        if differs:
            shown = ", ".join(differs[:8]) + (f" and {len(differs) - 8} more" if len(differs) > 8 else "")
            raise RunManifestError([f"{run_dir.name} was made with a different setup ({shown}). Results from "
                                    "different code, bank, prompts or providers are never mixed: use a new --run "
                                    "name (a new version, protocol section 4)."])
        return existing

    def check_condition(self, system: str, config: dict) -> None:
        known = self.data["conditions"].get(system)
        if known is None:
            self.data["conditions"][system] = config
        elif known != config:
            raise RunManifestError([f"{system} in {self.run_dir.name} was run with a different configuration "
                                    f"({', '.join(_diff(known, config)[:8])}). Use a new --run name."])

    def check_candidates(self, candidates: list[dict], digests: dict[str, str]) -> None:
        problems = []
        for c in candidates:
            digest, known = digests[c["id"]], self.data["candidates"].get(c["id"])
            if known is None:
                self.data["candidates"][c["id"]] = digest
            elif known != digest:
                problems.append(f"{c['id']}: candidate file differs from the one already interviewed in this run")
        if problems:
            raise RunManifestError(problems + ["Candidate keys are frozen for a run; use a new --run name."])

    def add_session(self, *, operator: str | None, share: str | None, systems: list[str],
                    candidate_ids: list[str], git: dict) -> None:
        self.data["sessions"].append({"started_at": now_iso(), "operator": operator, "share": share,
                                      "systems": list(systems), "candidates": list(candidate_ids), **git})

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------
# Failures and exclusions (protocol section 4)
# --------------------------------------------------------------------------

def record_failure(run_dir: Path, system: str, candidate_id: str, errors: list[str]) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / FAILURES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_iso(), "system": system, "candidate_id": candidate_id,
                            "errors": [e[:500] for e in errors]}) + "\n")


def failure_counts(run_dir: Path) -> dict[tuple[str, str], int]:
    path = Path(run_dir) / FAILURES
    counts: dict[tuple[str, str], int] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                key = (entry["system"], entry["candidate_id"])
                counts[key] = counts.get(key, 0) + 1
    return counts
