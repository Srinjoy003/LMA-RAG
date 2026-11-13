"""Step 7: the agent demonstration and the trusted-bank SQL verification tool. All offline."""

import copy
import json
import random
import subprocess
import sys

import pytest

from engine import agent
from engine.demo import ScriptedAgent, compare_drivers, summary_check
from engine.factory import build_components
from engine.run import run_interview
from engine.verification import applies_to, for_grader, summary, verify_sql_result
from engine_support import ROOT, real_bank
from fake_llm import FakeLLM
from part3.grader import PromptedGrader
from part4.candidates import load_candidates, truth_for
from part4.metrics import is_sql_read_code, verification_metrics
from shared.interfaces import SYSTEMS
from shared.stubs import RuleBasedRespondent

BANK = real_bank()
JOBS = {j["id"]: j for j in BANK.jobs}
SQL_CODE = [q for q in BANK.questions if q["skill"] == "sql" and q["type"] == "read-code"]
CANDIDATES = load_candidates(ROOT / "data" / "candidates")[:6]


class Spy(RuleBasedRespondent):
    """Records everything the candidate is shown."""

    def __init__(self, candidate):
        super().__init__(candidate)
        self.seen = []

    def reply(self, prompt, *, question, kind, turn):
        self.seen.append((prompt, copy.deepcopy(question)))
        return super().reply(prompt, question=question, kind=kind, turn=turn)


class CapturingGrader:
    """Wraps the stub grader and records the question object each grading call received."""

    def __init__(self, inner):
        self.inner, self.questions = inner, []

    def grade(self, question, answer, follow_ups=()):
        self.questions.append(question)
        return self.inner.grade(question, answer, follow_ups)


def interview(system, candidate, *, driver="code", probes=False, respondent=None, grader_spy=False):
    config = SYSTEMS[system].with_(driver=driver)
    components = build_components(config, live=False, root=ROOT)
    if grader_spy:
        components.grader = CapturingGrader(components.grader)
    if driver == "agent":
        components.llm = ScriptedAgent(probes=probes)
    respondent = respondent or RuleBasedRespondent(candidate)
    job = JOBS[candidate["job_id"]]
    result = run_interview(job, respondent, config, components, candidate_id=candidate["id"],
                           rng=random.Random(config.seed))
    result["truth"] = truth_for(candidate, job)
    return result, components


@pytest.fixture(scope="module")
def runs():
    """A2E and A2EV, code and agent driver, on six frozen candidates (stubs)."""
    out = {}
    for c in CANDIDATES:
        for system in ("A2E", "A2EV"):
            for driver in ("code", "agent"):
                out[(system, driver, c["id"])] = interview(system, c, driver=driver)[0]
    return out


# --------------------------------------------------------------------------
# The tool itself
# --------------------------------------------------------------------------

def test_every_sql_read_code_question_in_the_bank_verifies():
    assert len(SQL_CODE) == 18
    for q in SQL_CODE:
        evidence = verify_sql_result(q)
        assert evidence["status"] == "verified", (q["id"], evidence)
        assert evidence["row_count"] == len(q["expected_output"]) and evidence["matches_bank_output"]


def test_only_sql_read_code_bank_questions_qualify():
    explain = next(q for q in BANK.questions if q["skill"] == "sql" and q["type"] == "explain")
    python = next(q for q in BANK.questions if q["skill"] == "py" and q["type"] == "read-code")
    generated = {**SQL_CODE[0], "source": "generated-live"}
    for q in (explain, python, generated):
        assert not applies_to(q)
        with pytest.raises(ValueError):
            verify_sql_result(q)


def test_mismatch_and_error_are_reported_and_never_given_to_the_grader():
    wrong = {**SQL_CODE[0], "expected_output": [["not", "this"]]}
    broken = {**SQL_CODE[0], "code": "SELECT * FROM no_such_table"}
    assert verify_sql_result(wrong)["status"] == "mismatch"
    failed = verify_sql_result(broken)
    assert failed["status"] == "error" and "SQLite error" in failed["error"]
    assert for_grader(verify_sql_result(wrong)) is None and for_grader(failed) is None


def test_grader_view_has_rows_but_no_timing_and_summary_has_no_rows():
    evidence = verify_sql_result(SQL_CODE[0])
    view = for_grader(evidence)
    assert view["rows"] == evidence["rows"] and "elapsed_ms" not in view
    assert "rows" not in summary(evidence) and summary(evidence)["status"] == "verified"


# --------------------------------------------------------------------------
# Rules (code and agent driver)
# --------------------------------------------------------------------------

def test_tool_rules_are_enforced():
    from engine.selection import InterviewState
    from engine.tools import InterviewTools
    for system, sql_question, message in (("A2E", SQL_CODE[0], "does not use SQL verification"),
                                          ("A2EV", next(q for q in BANK.questions if q["skill"] == "py"),
                                           "not a SQL read-code question")):
        config = SYSTEMS[system]
        components = build_components(config, live=False, root=ROOT)
        state = InterviewState(JOBS["data-analyst"], config, components, [], random.Random(0))
        tools = InterviewTools(state, components.grader)
        state.pending = sql_question
        refused = tools.verify_sql_result(sql_question["id"])
        assert not refused["ok"] and message in refused["error"]
    state.pending = SQL_CODE[0]
    assert "only the open question" in tools.verify_sql_result("SQL-L1-001-NOT")["error"]
    assert tools.verify_sql_result(SQL_CODE[0]["id"])["ok"]
    assert "already verified" in tools.verify_sql_result(SQL_CODE[0]["id"])["error"]


def test_code_driver_verifies_every_sql_read_code_question_and_only_with_the_flag(runs):
    on = [t for (s, d, _), r in runs.items() if s == "A2EV" and d == "code" for t in r["turns"]]
    off = [t for (s, d, _), r in runs.items() if s == "A2E" and d == "code" for t in r["turns"]]
    sql_on = [t for t in on if t["question_type"] == "read-code" and t["skill"] == "sql"]
    assert sql_on, "the sample must include SQL read-code questions"
    assert all(t["verification"]["status"] == "verified" for t in sql_on)
    assert not any(t.get("verification") for t in on if t not in sql_on)
    assert not any(t.get("verification") for t in off)


def test_the_flag_changes_nothing_else_on_stubs(runs):
    """The stub grader ignores the evidence, so A2EV must ask, grade and decide exactly as A2E."""
    for c in CANDIDATES:
        a, b = runs[("A2E", "code", c["id"])], runs[("A2EV", "code", c["id"])]
        strip = lambda r: [{k: v for k, v in t.items() if k not in ("verification", "tool_calls")} for t in r["turns"]]
        assert strip(a) == strip(b)
        assert a["report"]["decision"] == b["report"]["decision"]


def test_the_grader_gets_the_rows_only_for_sql_read_code_questions():
    candidate = next(c for c in CANDIDATES
                     if any(t["question_type"] == "read-code" and t["skill"] == "sql"
                            for t in interview("A2EV", c)[0]["turns"]))
    _, on = interview("A2EV", candidate, grader_spy=True)
    _, off = interview("A2E", candidate, grader_spy=True)
    with_rows = [q for q in on.grader.questions if "verified_execution" in q]
    assert with_rows and all(applies_to(q) for q in with_rows)
    assert not any("verified_execution" in q for q in off.grader.questions)
    assert all(applies_to(q) or "verified_execution" not in q for q in on.grader.questions)


def test_the_candidate_never_sees_the_evidence():
    for c in CANDIDATES[:3]:
        spy_on, spy_off = Spy(c), Spy(c)
        interview("A2EV", c, respondent=spy_on)
        interview("A2E", c, respondent=spy_off)
        assert [p for p, _ in spy_on.seen] == [p for p, _ in spy_off.seen]      # identical prompts
        assert not any("verified_execution" in q for _, q in spy_on.seen if q)
        assert not any("VERIFIED EXECUTION" in p for p, _ in spy_on.seen)


def test_prompted_grader_adds_the_section_only_when_given():
    grader = PromptedGrader(FakeLLM([{"rationale": "ok", "score": 2}]), model="m", votes=1, misconceptions={})
    q = SQL_CODE[0]
    plain = grader.build_messages(q, "answer")
    verified = grader.build_messages({**q, "verified_execution": for_grader(verify_sql_result(q))}, "answer")
    assert "VERIFIED EXECUTION RESULT" not in plain[1]["content"]
    text = verified[1]["content"]
    assert f"exactly {len(q['expected_output'])} row(s)" in text
    assert plain[1]["content"].replace("\n\n", "") in text.replace("\n\n", "").replace(
        text[text.index("VERIFIED"):text.index("CANDIDATE'S ANSWER")].replace("\n\n", ""), "")


def test_agent_without_the_flag_sends_exactly_the_old_tools_and_prompt():
    seen = []

    class Recording(ScriptedAgent):
        def complete(self, messages, **kwargs):
            seen.append((messages[0]["content"], kwargs.get("tools")))
            return super().complete(messages, **kwargs)
    config = SYSTEMS["A2E"].with_(driver="agent", max_questions=2)
    components = build_components(config, live=False, root=ROOT)
    components.llm = Recording()
    run_interview(JOBS[CANDIDATES[0]["job_id"]], RuleBasedRespondent(CANDIDATES[0]), config, components,
                  candidate_id=CANDIDATES[0]["id"], rng=random.Random(0))
    assert all(prompt == agent.SYSTEM_PROMPT and tools == agent.TOOLS for prompt, tools in seen)
    assert agent.tools_for(SYSTEMS["A2EV"])[-1]["function"]["name"] == "verify_sql_result"


def test_obedient_agent_matches_the_code_driver_with_verification(runs):
    for c in CANDIDATES:
        for system in ("A2E", "A2EV"):
            comparison = compare_drivers(runs[(system, "agent", c["id"])], runs[(system, "code", c["id"])])
            assert comparison["same_questions"] and comparison["same_decision"] and comparison["same_levels"]
        agent_turns = runs[("A2EV", "agent", c["id"])]["turns"]
        code_turns = runs[("A2EV", "code", c["id"])]["turns"]
        assert [t.get("verification", {}).get("status") for t in agent_turns] == \
            [t.get("verification", {}).get("status") for t in code_turns]


def test_rule_breaking_calls_are_refused_and_change_nothing(runs):
    candidate = next(c for c in CANDIDATES if any(t.get("verification") for t in runs[("A2EV", "code", c["id"])]["turns"]))
    probed, _ = interview("A2EV", candidate, driver="agent", probes=True)
    plain = runs[("A2EV", "agent", candidate["id"])]
    info = probed["agent"]
    assert info["refused_calls"] >= 5 and info["invalid_calls"] == 1
    errors = [t["error"] for t in info["tool_timings"] if not t["ok"]]
    for expected in ("stop rule", "the rules chose question", "nothing has been graded", "still open",
                     "needs verify_sql_result before grade_answer", "unknown tool"):
        assert any(expected in e for e in errors), expected
    assert compare_drivers(probed, plain)["same_questions"]
    assert probed["report"]["decision"] == plain["report"]["decision"] and probed["stop_reason"] == plain["stop_reason"]


# --------------------------------------------------------------------------
# Summary check and metrics
# --------------------------------------------------------------------------

LEVELS = {"py": {"level": 2}, "sql": {"level": 1}, "ds": {"level": 3}}


@pytest.mark.parametrize("decision, summary_text, consistent", [
    ("needs_review", "Python level 2, SQL level 1. The evidence is not conclusive.", True),
    ("needs_review", "A strong fit for the role.", False),
    ("fit", "The candidate is a fit; SQL level 1.", True),
    ("fit", "SQL level 3 and data structures level 3.", False),
    ("not_fit", "We should hire this candidate.", False),
    ("not_fit", "Not fit: SQL is below the job minimum (level 2).", True),
])
def test_summary_check(decision, summary_text, consistent):
    assert summary_check({"decision": decision, "levels": LEVELS, "summary": summary_text})["consistent"] is consistent


def test_verification_metrics(runs):
    results = copy.deepcopy([r for (s, d, _), r in runs.items() if s == "A2EV" and d == "code"])
    assert verification_metrics(results)["sql_read_code_grading_accuracy"] is None     # no plans merged yet
    sql_turns = [t for r in results for t in r["turns"] if is_sql_read_code(t)]
    for i, t in enumerate(sql_turns):        # plans as the runner merges them: one wrong grade
        t["truth"] = {"expected_score": t["grade"]["score"] if i else (t["grade"]["score"] + 1) % 3}
    m = verification_metrics(results)
    assert m["sql_read_code_questions"] == m["verified_questions"] == len(sql_turns) > 0
    assert m["verification_failures"] == 0
    assert m["sql_read_code_grading_accuracy"] == pytest.approx((len(sql_turns) - 1) / len(sql_turns))
    old = copy.deepcopy(results)                  # results from before Step 7: no question_type
    for r in old:
        for t in r["turns"]:
            t.pop("question_type")
    assert [is_sql_read_code(t) for r in old for t in r["turns"]] == \
        [is_sql_read_code(t) for r in results for t in r["turns"]]


# --------------------------------------------------------------------------
# Demo command
# --------------------------------------------------------------------------

def test_demo_cli_runs_offline_and_never_overwrites(tmp_path):
    out = tmp_path / "demo.json"
    command = [sys.executable, str(ROOT / "scripts/agent_demo.py"), "--system", "A2EV", "--candidate", "CAND-002",
               "--probe-rules", "--out", str(out)]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "REFUSED" in proc.stdout and "verify_sql_result" in proc.stdout and "consistent" in proc.stdout
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["comparison"]["same_questions"] and record["summary_check"]["consistent"]
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "exists" in again.stderr
