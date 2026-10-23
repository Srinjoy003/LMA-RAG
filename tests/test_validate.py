import subprocess
import sys

from part1.bank import load_bank
from part1.validate import Rules, validate_bank

from conftest import fake_embed
from fixtures.make_bank import PROJECT_ROOT, make_question, write_bank


def q(bank, qid):
    return bank.get_question(qid)


def run(bank, **kw):
    kw.setdefault("execute", False)
    return validate_bank(bank, **kw)


def messages(report):
    return "\n".join(report.errors)


# ---- the passing bank --------------------------------------------------
def test_valid_bank_passes_including_execution(bank):
    report = validate_bank(bank, execute=True)
    assert report.errors == []
    assert report.warnings == []


def test_missing_files_are_empty_not_a_crash(tmp_path):
    bank = load_bank(tmp_path / "nothing-here")
    assert bank.questions == [] and bank.misconceptions == [] and bank.jobs == []
    assert bank.get_question("PY-L1-001") is None


def test_skill_without_data_is_a_warning_only(tmp_path):
    root = write_bank(tmp_path / "b", skills=("py",))
    report = run(load_bank(root))
    assert report.errors == []
    assert any("sql" in w and "no questions" in w for w in report.warnings)


def test_skill_filter_only_checks_that_skill(bank):
    q(bank, "SQL-L1-001")["level"] = 7   # broken, but in another skill
    assert run(bank, skill="py").errors == []
    assert run(bank, skill="sql").errors


# ---- a. schema ---------------------------------------------------------
def test_schema_error_on_question(bank):
    del q(bank, "PY-L1-001")["model_answer"]
    assert "[schema] PY-L1-001" in messages(run(bank))


def test_schema_error_read_code_without_code(bank):
    del q(bank, "PY-L1-007")["code"]
    assert "[schema] PY-L1-007" in messages(run(bank))


def test_schema_error_on_misconception(bank):
    del bank.misconception_by_id["py.alpha.wrong-idea"]["description"]
    assert "[schema] py.alpha.wrong-idea" in messages(run(bank))


def test_non_object_item_is_reported(bank):
    bank.questions.append("oops")
    assert "not a JSON object" in messages(run(bank))


def test_bad_json_file_is_reported(bank_dir):
    (bank_dir / "data" / "questions" / "py.json").write_text("[ not json", encoding="utf-8")
    report = run(load_bank(bank_dir))
    assert any(e.startswith("[load]") for e in report.errors)


# ---- b. ids --------------------------------------------------------------
def test_duplicate_question_id(bank):
    bank.questions.append(dict(q(bank, "PY-L1-001")))
    assert "duplicate question id 'PY-L1-001'" in messages(run(bank))


def test_duplicate_misconception_id(bank):
    bank.misconceptions.append(dict(bank.misconceptions[0]))
    assert "duplicate misconception id" in messages(run(bank))


def test_question_id_prefix_must_match_skill(bank):
    q(bank, "PY-L1-001")["skill"] = "sql"
    assert "does not match skill" in messages(run(bank))


def test_question_id_level_must_match_level(bank):
    q(bank, "PY-L1-001")["level"] = 2
    assert "does not match level" in messages(run(bank))


def test_misconception_id_must_match_skill(bank):
    bank.misconception_by_id["py.alpha.wrong-idea"]["skill"] = "sql"
    assert "id starts with 'py' but skill is 'sql'" in messages(run(bank))


def test_misconception_id_must_match_topic(bank):
    bank.misconception_by_id["py.alpha.wrong-idea"]["topic"] = "py-beta"
    assert "does not match topic" in messages(run(bank))


# ---- c. references ---------------------------------------------------------
def test_unknown_topic(bank):
    q(bank, "PY-L1-001")["topics"] = ["py-nonexistent"]
    assert "'py-nonexistent' is not in topics.json" in messages(run(bank))


def test_topic_of_other_skill(bank):
    q(bank, "PY-L1-001")["topics"] = ["sql-alpha"]
    assert "belongs to skill 'sql', not 'py'" in messages(run(bank))


def test_unknown_misconception(bank):
    q(bank, "PY-L1-001")["misconceptions"] = ["py.alpha.does-not-exist"]
    assert "does not exist" in messages(run(bank))


def test_misconception_of_other_skill(bank):
    q(bank, "PY-L1-001")["misconceptions"] = ["sql.alpha.wrong-idea"]
    assert "has skill 'sql', not 'py'" in messages(run(bank))


def test_misconception_topic_must_exist(bank):
    bank.misconception_by_id["py.alpha.wrong-idea"]["topic"] = "py-ghost"
    assert "topic 'py-ghost' is not in topics.json" in messages(run(bank))


def test_job_required_topic_must_exist(bank):
    bank.jobs[0]["required_topics"].append("py-ghost")
    assert "job job-1: required topic 'py-ghost'" in messages(run(bank))


# ---- d. coverage ---------------------------------------------------------------
def test_levels_of_a_skill_must_be_the_same_size(bank):
    extra = make_question("py", 1, 1)
    extra["id"] = "PY-L1-011"
    bank.questions.append(extra)
    assert "py: levels 1-3 have 11, 10, 10 questions" in messages(run(bank))


def test_too_few_questions_per_level(bank):
    bank.questions = [x for x in bank.questions if not x["id"].endswith("-010")]
    msg = messages(run(bank))
    assert "py level 2: 9 questions, need at least 10" in msg and "levels 1-3 have" not in msg


def test_a_bigger_balanced_bank_passes(bank):
    for level in (1, 2, 3):
        for new, source in zip(range(11, 16), (1, 2, 3, 7, 8)):     # 3 explain + 2 read-code
            bank.questions.append({**make_question("py", level, source), "id": f"PY-L{level}-{new:03d}"})
    assert "[coverage] py" not in messages(run(bank))


def test_wrong_explain_read_code_split(bank):
    q(bank, "PY-L1-007")["type"] = "explain"
    assert "py level 1: 7 explain and 3 read-code questions, expected 6 and 4" in messages(run(bank))


def test_too_few_topics_per_level(bank):
    for x in bank.questions:
        if x["skill"] == "py" and x["level"] == 1:
            x["topics"] = ["py-alpha", "py-beta"]
    assert "py level 1: 2 distinct topics" in messages(run(bank))


def test_topic_with_too_few_questions(bank):
    # py-epsilon is not a job-required topic, so only the "at least 2" rule applies
    for x in bank.questions:
        if x["skill"] == "py" and "py-epsilon" in x["topics"]:
            x["topics"] = ["py-alpha"]
    msg = messages(run(bank))
    assert "topic py-epsilon: 0 questions, need at least 2" in msg
    assert "job-required topic py-epsilon" not in msg


def test_required_topic_with_too_few_questions(bank):
    # leave exactly 2 questions on py-alpha (required by job-1): enough for "2", not for "3"
    holders = [x for x in bank.questions if x["skill"] == "py" and "py-alpha" in x["topics"]]
    for x in holders[2:]:
        x["topics"] = ["py-beta"]
    msg = messages(run(bank))
    assert "job-required topic py-alpha: 2 questions, need at least 3" in msg
    assert "topic py-alpha: 2 questions, need at least 2" not in msg


def test_misconception_referenced_too_rarely(bank):
    for x in bank.questions:
        if x["skill"] == "py":
            x["misconceptions"] = ["py.alpha.wrong-idea"]
    assert "misconception py.epsilon.wrong-idea: referenced by 0 questions" in messages(run(bank))


def test_rules_are_configurable(bank):
    report = run(bank, rules=Rules(min_questions_per_level=11))
    assert "need at least 11" in messages(report)


# ---- e. warnings -------------------------------------------------------------------
def test_empty_job_description_is_a_warning(bank):
    bank.jobs[0]["description"] = ""
    report = run(bank)
    assert report.errors == []
    assert any("description is empty" in w for w in report.warnings)


def test_read_code_without_expected_output_is_a_warning(bank):
    del q(bank, "PY-L1-007")["expected_output"]
    report = run(bank)
    assert report.errors == []
    assert any("PY-L1-007" in w and "expected_output" in w for w in report.warnings)


# ---- f. execution -------------------------------------------------------------------
def test_wrong_python_output_is_an_error(bank):
    q(bank, "PY-L1-007")["expected_output"] = "999\n"
    assert "[exec] PY-L1-007: output mismatch" in messages(validate_bank(bank, execute=True))


def test_wrong_sql_output_is_an_error(bank):
    q(bank, "SQL-L1-007")["expected_output"] = [[1, "Asha"]]
    assert "[exec] SQL-L1-007: result mismatch" in messages(validate_bank(bank, execute=True))


def test_no_exec_skips_execution(bank):
    q(bank, "PY-L1-007")["expected_output"] = "999\n"
    assert run(bank, execute=False).errors == []


def test_python_timeout_is_an_error(bank):
    q(bank, "PY-L1-007")["code"] = "while True: pass"
    report = validate_bank(bank, skill="py", execute=True, timeout=1)
    assert "timed out" in messages(report)


# ---- g. duplicates ---------------------------------------------------------------------
def test_dupes_flagged_when_requested(bank):
    a, b = q(bank, "PY-L1-001"), q(bank, "PY-L2-002")
    for key in ("question", "topics", "key_points"):
        b[key] = a[key]
    report = run(bank, dupes=True, embed_fn=fake_embed)
    assert report.errors == []
    assert any("PY-L1-001" in w and "PY-L2-002" in w and "[dupes]" in w for w in report.warnings)
    assert not any("[dupes]" in w for w in run(bank).warnings)


# ---- the script -------------------------------------------------------------------------
def run_script(root, *args):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "validate_bank.py"),
                           "--root", str(root), *args], capture_output=True, text=True)


def test_script_exit_codes(bank_dir):
    ok = run_script(bank_dir)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "0 errors" in ok.stdout

    path = bank_dir / "data" / "questions" / "py.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"PY-L1-001"', '"PY-L1-002"', 1), encoding="utf-8")
    bad = run_script(bank_dir, "--no-exec", "--skill", "py")
    assert bad.returncode == 1
    assert "duplicate question id" in bad.stdout
