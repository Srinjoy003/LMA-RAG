"""`verify_sql_result`: run a SQL read-code question's own query on its own tables (Step 7).

Only trusted bank content is executed: the query and tables of the question the rules chose,
looked up by its id, through the existing runner in part1/execute.py (an in-memory SQLite
database with a time limit). Nothing the candidate wrote is ever executed; running candidate
code would need a separate isolation design.

The evidence goes to the grader only. It is never part of a prompt shown to the candidate,
and the agent sees only its status, never the rows.

Statuses:
    verified   the query ran and its rows equal the bank's checked `expected_output`
    mismatch   the query ran but disagrees with `expected_output`: the bank needs a look, so the
               rows are NOT given to the grader (they would contradict the model answer)
    error      the query failed (SQLite error or time limit); nothing is given to the grader
"""

from __future__ import annotations

import time

from part1.execute import rows_equal, run_sql

MAX_ROWS_TO_GRADER = 50      # rows shown to the grader; the row count is always exact


def applies_to(question: dict) -> bool:
    """A SQL read-code question from the bank: it has a query whose output is checked."""
    return (question.get("skill") == "sql" and question.get("type") == "read-code"
            and bool(question.get("code")) and question.get("expected_output") is not None
            and question.get("source") != "generated-live")


def verify_sql_result(question: dict) -> dict:
    """Execute the question's own query on its own tables and describe the result."""
    if not applies_to(question):
        raise ValueError(f"{question.get('id')} is not a SQL read-code question from the bank")
    started = time.perf_counter()
    evidence = {"tool": "verify_sql_result", "question_id": question["id"]}
    try:
        rows, ordered = run_sql(question["code"], question.get("tables"))
    except RuntimeError as e:
        evidence.update(status="error", error=str(e)[:300])
    else:
        matches = rows_equal(rows, question["expected_output"], ordered)
        evidence.update(status="verified" if matches else "mismatch", row_count=len(rows), ordered=ordered,
                        rows=rows[:MAX_ROWS_TO_GRADER], rows_truncated=len(rows) > MAX_ROWS_TO_GRADER,
                        matches_bank_output=matches)
    evidence["elapsed_ms"] = round(1000 * (time.perf_counter() - started), 3)
    return evidence


def for_grader(evidence: dict) -> dict | None:
    """The part of the evidence the grader sees: only a verified result, and nothing that
    changes between runs (no timing), so the grader's prompt and cache key are stable."""
    if evidence.get("status") != "verified":
        return None
    return {"row_count": evidence["row_count"], "ordered": evidence["ordered"], "rows": evidence["rows"],
            "rows_truncated": evidence["rows_truncated"]}


def summary(evidence: dict) -> dict:
    """What the agent and the saved turn show: status and counts, never the rows."""
    return {k: evidence[k] for k in ("question_id", "status", "row_count", "matches_bank_output", "error",
                                     "elapsed_ms") if k in evidence}
