"""Sampling, blinding, reproducibility and preservation of independent labels."""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from part4.difficulty import prepare_difficulty


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "bank"
    folder = root / "data/questions"
    folder.mkdir(parents=True)
    for skill in ("py", "sql", "ds"):
        questions = [{"id": f"{skill.upper()}-L{i % 3 + 1}-{i:03d}", "skill": skill,
                      "level": i % 3 + 1, "status": "checked", "topics": ["SECRET-TOPIC"],
                      "question": f"Explain example {i}.", "model_answer": "Reference answer.",
                      "key_points": ["Explain the reasoning."], "code": "print('```')",
                      "tables": [{"name": "example", "columns": ["name", "value"],
                                  "rows": [["a", None], ["b", 2]]}]} for i in range(10)]
        (folder / f"{skill}.json").write_text(json.dumps(questions))
    return root


def test_balanced_sample_and_blind_packet(source, tmp_path):
    output = tmp_path / "packet"
    result = prepare_difficulty(source, output)
    assert result["items"] == 21
    key = json.loads((output / "private/key.json").read_text())
    assert Counter(k["skill"] for k in key) == {"py": 7, "sql": 7, "ds": 7}
    assert len({k["question_id"] for k in key}) == 21
    assert [k["item"] for k in key] == [f"D{i:02d}" for i in range(1, 22)]
    text = (output / "packet.md").read_text()
    for k in key:
        assert k["question_id"] not in text
        assert f"## {k['item']}\n" in text
    assert "SECRET-TOPIC" not in text
    assert "assigned_level" not in text
    assert "Reference answer." in text and "Explain the reasoning." in text
    assert "a | NULL" in text and "b | 2" in text
    assert "````python" in text  # nested triple backticks cannot close the code block
    assert "key" not in result


def test_raters_get_matching_but_separate_blank_sheets(source, tmp_path):
    output = tmp_path / "packet"
    prepare_difficulty(source, output)
    sheets = {}
    for name in ("souradeep", "nilkanta"):
        with (output / f"ratings/{name}.csv").open(newline="") as f:
            sheets[name] = list(csv.DictReader(f))
        assert len(sheets[name]) == 21
        assert all(r["rater"] == name and r["level"] == r["note"] == "" for r in sheets[name])
        assert set(sheets[name][0]) == {"item", "rater", "level", "note"}
    assert [r["item"] for r in sheets["souradeep"]] == [r["item"] for r in sheets["nilkanta"]]


def test_reproducibility_and_source_order_independence(source, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    prepare_difficulty(source, first)
    for path in (source / "data/questions").glob("*.json"):
        path.write_text(json.dumps(list(reversed(json.loads(path.read_text())))))
    prepare_difficulty(source, second)
    for name in ("packet.md", "private/key.json", "ratings/souradeep.csv", "ratings/nilkanta.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    prepare_difficulty(source, tmp_path / "other", seed=1402)
    assert (first / "private/key.json").read_bytes() != (tmp_path / "other/private/key.json").read_bytes()


def test_manifest_hashes_actual_files_and_records_settings(source, tmp_path):
    output = tmp_path / "packet"
    prepare_difficulty(source, output)
    manifest = json.loads((output / "private/manifest.json").read_text())
    assert manifest["seed"] == 1401 and manifest["per_skill"] == 7
    assert len(manifest["generator_sha256"]) == 64
    for name, digest in manifest["public_files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["source_files_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((output / "private/key.json").read_bytes()).hexdigest() == manifest["key_sha256"]


def test_refuses_to_overwrite_completed_labels(source, tmp_path):
    output = tmp_path / "packet"
    prepare_difficulty(source, output)
    sheet = output / "ratings/souradeep.csv"
    sheet.write_text("completed ratings must survive\n")
    original_key = (output / "private/key.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_difficulty(source, output)
    assert sheet.read_text() == "completed ratings must survive\n"
    assert (output / "private/key.json").read_bytes() == original_key


@pytest.mark.parametrize("per_skill", [0, True, 11])
def test_invalid_sample_fails_before_creating_output(source, tmp_path, per_skill):
    with pytest.raises(ValueError):
        prepare_difficulty(source, tmp_path / "packet", per_skill=per_skill)
    assert not (tmp_path / "packet").exists()


def test_draft_bank_is_rejected(source, tmp_path):
    path = source / "data/questions/py.json"
    questions = json.loads(path.read_text()); questions[0]["status"] = "draft"
    path.write_text(json.dumps(questions))
    with pytest.raises(ValueError, match="checked"):
        prepare_difficulty(source, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_ids_embedded_in_display_text_are_not_leaked(source, tmp_path):
    path = source / "data/questions/py.json"
    questions = json.loads(path.read_text())
    for q in questions:
        q["question"] += " See " + q["id"]
    path.write_text(json.dumps(questions))
    with pytest.raises(ValueError, match="question ID"):
        prepare_difficulty(source, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_cli_prints_only_public_summary_and_generated_folder_ignores_itself(source, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_difficulty.py"
    output = tmp_path / "packet"
    proc = subprocess.run([sys.executable, str(script), "--root", str(source), "--out", str(output)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "21 items (7 per skill)" in proc.stdout
    assert not any(prefix in proc.stdout for prefix in ("PY-L", "SQL-L", "DS-L"))
    assert (output / ".gitignore").read_text() == "*\n"
    repeated = subprocess.run([sys.executable, str(script), "--root", str(source), "--out", str(output)],
                              capture_output=True, text=True)
    assert repeated.returncode != 0 and "Preparation failed" in repeated.stderr
