"""scripts/run_rotating.py: moves each role to its next key only when that key is used up."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_rotating", ROOT / "scripts" / "run_rotating.py")
rot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rot)

ARGS = ["--grader-keys", "2", "3", "main", "--simulator-keys", "main", "2",
        "--grader", "gemini", "--simulator", "groq", "--run", "test"]


@pytest.fixture
def keys(monkeypatch):
    for var, value in {"GEMINI_API_KEY": "g-paid", "GEMINI_API_KEY_2": "g2", "GEMINI_API_KEY_3": "g3",
                       "GROQ_API_KEY": "q1", "GROQ_API_KEY_2": "q2"}.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(rot, "load_env", lambda: None)
    monkeypatch.setattr(rot.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_rotating.py", *ARGS])


def drive(monkeypatch, quota_stops: int, used_up: set):
    """Fake runs: the first `quota_stops` runs stop on quota; `used_up` keys fail the check."""
    used = []

    def fake_run(rest, gemini_key, groq_key):
        used.append((gemini_key, groq_key))
        return len(used) <= quota_stops, ""

    monkeypatch.setattr(rot, "run_once", fake_run)
    monkeypatch.setattr(rot, "key_used_up", lambda provider, key, model: key in used_up)
    return rot.main(), used


def test_free_grader_keys_come_before_the_paid_one(keys, monkeypatch):
    code, used = drive(monkeypatch, quota_stops=2, used_up={"g2", "g3"})
    assert code == 0
    assert [g for g, _ in used] == ["g2", "g3", "g-paid"]
    assert {q for _, q in used} == {"q1"}            # the simulator key was never used up


def test_only_the_used_up_role_moves_on(keys, monkeypatch):
    code, used = drive(monkeypatch, quota_stops=1, used_up={"q1"})
    assert code == 0
    assert used == [("g2", "q1"), ("g2", "q2")]


def test_stops_when_a_role_has_no_keys_left(keys, monkeypatch):
    code, used = drive(monkeypatch, quota_stops=5, used_up={"q1", "q2"})
    assert code == 3
    assert used == [("g2", "q1"), ("g2", "q2")]


def test_a_short_term_limit_pauses_then_rests_the_grader_key(keys, monkeypatch):
    code, used = drive(monkeypatch, quota_stops=3, used_up=set())
    assert code == 0
    assert [g for g, _ in used] == ["g2", "g2", "g2", "g3"]


def test_by_candidate_runs_each_candidate_and_keeps_the_key_state(keys, monkeypatch, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("CAND-001\nCAND-002\nCAND-003\n")
    monkeypatch.setattr(sys, "argv", ["run_rotating.py", *ARGS, "--by-candidate", str(ids)])
    calls = []

    def fake_run(rest, gemini_key, groq_key):
        calls.append((rest[rest.index("--ids") + 1], groq_key))
        return len(calls) == 2, ""        # CAND-002's first attempt stops on quota

    monkeypatch.setattr(rot, "run_once", fake_run)
    monkeypatch.setattr(rot, "key_used_up", lambda provider, key, model: key == "q1")
    assert rot.main() == 0
    # CAND-002 resumes on the next simulator key, and CAND-003 stays on it
    assert calls == [("CAND-001", "q1"), ("CAND-002", "q1"), ("CAND-002", "q2"), ("CAND-003", "q2")]


def test_the_quota_error_names_the_used_up_role(keys, monkeypatch):
    """The Groq error names the model: move the simulator on even if a test request still passes."""
    used = []

    def fake_run(rest, gemini_key, groq_key):
        used.append((gemini_key, groq_key))
        if len(used) == 1:
            return True, "LLMQuotaError: Rate limit reached for model `openai/gpt-oss-120b` ... (TPD)"
        return False, ""

    monkeypatch.setattr(rot, "run_once", fake_run)
    monkeypatch.setattr(rot, "key_used_up", lambda provider, key, model: False)
    monkeypatch.setattr(sys, "argv", ["run_rotating.py", *ARGS, "--simulator-model", "openai/gpt-oss-120b"])
    assert rot.main() == 0
    assert used == [("g2", "q1"), ("g2", "q2")]
