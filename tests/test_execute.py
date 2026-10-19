from part1.execute import (check_python, check_question, check_sql, has_top_level_order_by,
                           normalise_text, split_statements)

PEOPLE = [{"name": "t", "columns": ["id", "v"], "rows": [[1, 10], [2, None], [3, 30]]}]


# ---- python ----------------------------------------------------------------
def test_python_match():
    assert check_python("print('a')\nprint('b')", "a\nb\n") is None


def test_python_trailing_whitespace_is_ignored():
    assert check_python("print('a   ')\nprint()\nprint()", "a\n") is None
    assert normalise_text("x  \ny\t\n\n\n") == "x\ny"


def test_python_leading_whitespace_matters():
    assert check_python("print(' a')", "a") is not None


def test_python_mismatch_message():
    msg = check_python("print(1)", "2")
    assert "mismatch" in msg and "'1'" in msg


def test_python_exception_is_reported():
    msg = check_python("raise ValueError('boom')", "")
    assert "exited with status" in msg and "boom" in msg


def test_python_timeout():
    assert "timed out" in check_python("while True: pass", "", timeout=1)


def test_python_expected_must_be_string():
    assert "must be a string" in check_python("print(1)", [[1]])


def test_python_runs_with_this_interpreter_and_pandas():
    code = "import pandas as pd\nprint(pd.Series([1, 2]).sum())"
    assert check_python(code, "3") is None


# ---- sql: ordered vs multiset -------------------------------------------------
def test_sql_without_order_by_compares_as_multiset():
    code = "SELECT id FROM t"
    assert check_sql(code, PEOPLE, [[3], [1], [2]]) is None


def test_sql_multiset_still_counts_duplicates():
    code = "SELECT 1 FROM t"
    assert check_sql(code, PEOPLE, [[1], [1], [1]]) is None
    assert check_sql(code, PEOPLE, [[1], [1]]) is not None


def test_sql_with_order_by_compares_in_order():
    code = "SELECT id FROM t ORDER BY id DESC"
    assert check_sql(code, PEOPLE, [[3], [2], [1]]) is None
    assert "in order" in check_sql(code, PEOPLE, [[1], [2], [3]])


def test_order_by_inside_window_or_subquery_is_not_top_level():
    assert not has_top_level_order_by("SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t")
    assert not has_top_level_order_by("SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT 2)")
    assert not has_top_level_order_by("SELECT 'order by' FROM t -- order by id")
    assert has_top_level_order_by("SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t ORDER BY id")
    assert has_top_level_order_by("select id from t order\n  by id")


def test_window_order_by_does_not_force_row_order():
    code = "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t"
    assert check_sql(code, PEOPLE, [[3, 3], [1, 1], [2, 2]]) is None


# ---- sql: null, numbers ---------------------------------------------------------
def test_sql_null_matches_none():
    code = "SELECT id, v FROM t"
    assert check_sql(code, PEOPLE, [[1, 10], [2, None], [3, 30]]) is None
    assert check_sql(code, PEOPLE, [[1, 10], [2, 0], [3, 30]]) is not None


def test_sql_null_is_not_empty_string_or_zero():
    code = "SELECT v FROM t WHERE id = 2"
    assert check_sql(code, PEOPLE, [[""]]) is not None
    assert check_sql(code, PEOPLE, [[0]]) is not None
    assert check_sql(code, PEOPLE, [[None]]) is None


def test_sql_int_equals_float():
    assert check_sql("SELECT 2", None, [[2.0]]) is None
    assert check_sql("SELECT 2.0", None, [[2]]) is None
    assert check_sql("SELECT 1.0 * 10 / 4", None, [[2.5]]) is None
    assert check_sql("SELECT 2", None, [[3]]) is not None


def test_sql_string_does_not_equal_number():
    assert check_sql("SELECT '2'", None, [[2]]) is not None


# ---- sql: statements --------------------------------------------------------------
def test_sql_multiple_statements_use_last_result_set():
    code = "CREATE TABLE x (a); INSERT INTO x VALUES (5); SELECT a FROM x; DELETE FROM x;"
    assert check_sql(code, None, [[5]]) is None


def test_sql_semicolon_inside_string_does_not_split():
    assert split_statements("SELECT ';' AS s; SELECT 2;") == ["SELECT ';' AS s;", "SELECT 2;"]
    assert check_sql("SELECT ';'", None, [[";"]]) is None


def test_sql_no_rows_statement_is_a_problem():
    assert "no statement returned rows" in check_sql("CREATE TABLE y (a)", None, [])


def test_sql_error_is_reported():
    assert "SQLite error" in check_sql("SELECT nope FROM t", PEOPLE, [])


def test_sql_expected_must_be_rows():
    assert "list of rows" in check_sql("SELECT 1", None, "1")


def test_sql_empty_result_matches_empty_expected():
    assert check_sql("SELECT id FROM t WHERE id > 99", PEOPLE, []) is None


# ---- dispatch ------------------------------------------------------------------------
def test_check_question_dispatches_on_skill():
    sql_q = {"skill": "sql", "code": "SELECT COUNT(*) FROM t", "tables": PEOPLE, "expected_output": [[3]]}
    py_q = {"skill": "ds", "code": "print(len([1, 2, 3]))", "expected_output": "3\n"}
    assert check_question(sql_q) is None
    assert check_question(py_q) is None
