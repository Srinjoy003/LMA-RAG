"""Run read-code questions and compare the real output with `expected_output`.

- py / ds: the code runs in a subprocess with the same interpreter; stdout is compared as text.
- sql: the tables are loaded into an in-memory SQLite database; result rows are compared.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter

TIMEOUT_SECONDS = 10


# --------------------------------------------------------------------------
# Python / data structures
# --------------------------------------------------------------------------
def normalise_text(text: str) -> str:
    """Strip trailing whitespace on each line and trailing newlines at the end."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).rstrip("\n")


def run_python(code: str, timeout: float = TIMEOUT_SECONDS) -> str:
    """Run `code` in a fresh subprocess and return its stdout.

    Raises RuntimeError on a non-zero exit code or a timeout.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, encoding="utf-8",
                timeout=timeout, cwd=tmp, env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"code timed out after {timeout} s") from None
    if proc.returncode != 0:
        last = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        raise RuntimeError(f"code exited with status {proc.returncode}: {last[0]}")
    return proc.stdout


def check_python(code: str, expected: str, timeout: float = TIMEOUT_SECONDS) -> str | None:
    """Return None if the output matches, otherwise a message."""
    if not isinstance(expected, str):
        return "expected_output must be a string for py/ds questions"
    try:
        actual = run_python(code, timeout)
    except RuntimeError as e:
        return str(e)
    if normalise_text(actual) != normalise_text(expected):
        return f"output mismatch: expected {normalise_text(expected)!r}, got {normalise_text(actual)!r}"
    return None


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------
def split_statements(sql: str) -> list[str]:
    """Split SQL text into statements (semicolons inside strings/comments are respected)."""
    statements, start = [], 0
    for i, ch in enumerate(sql):
        if ch == ";":
            chunk = sql[start:i + 1]
            if sqlite3.complete_statement(chunk):
                statements.append(chunk.strip())
                start = i + 1
    rest = sql[start:].strip()
    if rest:
        statements.append(rest)
    return statements


def _strip_sql(sql: str) -> str:
    """Remove comments and string literals so keywords can be searched safely."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    return sql


def has_top_level_order_by(statement: str) -> bool:
    """True if the statement has an ORDER BY outside any parentheses."""
    text = _strip_sql(statement)
    depth = 0
    for match in re.finditer(r"[()]|\border\s+by\b", text, flags=re.I):
        token = match.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            return True
    return False


def run_sql(code: str, tables: list[dict] | None, timeout: float = TIMEOUT_SECONDS):
    """Execute `code` against the question's tables.

    Returns (rows, ordered) for the last statement that returned rows, where
    `ordered` says whether that statement has a top-level ORDER BY.
    Raises RuntimeError if no statement returns rows or SQLite fails.
    """
    conn = sqlite3.connect(":memory:")
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
    try:
        for t in tables or []:
            cols = ", ".join(f'"{c}"' for c in t["columns"])
            conn.execute(f'CREATE TABLE "{t["name"]}" ({cols})')
            marks = ", ".join("?" for _ in t["columns"])
            conn.executemany(f'INSERT INTO "{t["name"]}" VALUES ({marks})', t["rows"])
        result = None
        for stmt in split_statements(code):
            cur = conn.execute(stmt)
            if cur.description is not None:
                result = ([list(r) for r in cur.fetchall()], has_top_level_order_by(stmt))
        if result is None:
            raise RuntimeError("no statement returned rows")
        return result
    except sqlite3.Error as e:
        raise RuntimeError(f"SQLite error: {e}") from None
    finally:
        conn.close()


def _canon(value):
    """Canonical form of one cell: None, numbers (2 == 2.0) and strings."""
    if value is None:
        return ("null",)
    if isinstance(value, (int, float)):
        return ("num", round(float(value), 6))
    return ("str", str(value))


def _canon_row(row) -> tuple:
    return tuple(_canon(v) for v in row)


def rows_equal(actual: list, expected: list, ordered: bool) -> bool:
    a = [_canon_row(r) for r in actual]
    e = [_canon_row(r) for r in expected]
    return a == e if ordered else Counter(a) == Counter(e)


def check_sql(code: str, tables: list[dict] | None, expected,
              timeout: float = TIMEOUT_SECONDS) -> str | None:
    if not isinstance(expected, list):
        return "expected_output must be a list of rows for sql questions"
    try:
        actual, ordered = run_sql(code, tables, timeout)
    except RuntimeError as e:
        return str(e)
    if not rows_equal(actual, expected, ordered):
        how = "in order" if ordered else "as a multiset"
        return f"result mismatch ({how}): expected {expected!r}, got {actual!r}"
    return None


# --------------------------------------------------------------------------
def check_question(q: dict, timeout: float = TIMEOUT_SECONDS) -> str | None:
    """Run one read-code question. Returns None when it matches, else a message."""
    if q["skill"] == "sql":
        return check_sql(q["code"], q.get("tables"), q["expected_output"], timeout)
    return check_python(q["code"], q["expected_output"], timeout)
