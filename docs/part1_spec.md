# Part 1 spec: question bank and job data

Decisions 1–4 are frozen: Parts 2–4 build on them. Change them only with the whole team.

## Files

| File | Contents |
| --- | --- |
| `data/topics.json` | The 27 topic tags (9 per skill), with the levels each topic usually appears at |
| `data/jobs.json` | The 3 jobs: minimum level per skill and required topics (descriptions still to write) |
| `data/questions/{py,sql,ds}.json` | The bank: one list of questions per skill |
| `data/misconceptions/{py,sql,ds}.json` | The global misconception list, one file per skill |
| `index/` | The search index, built by `scripts/build_index.py` |
| `schemas/question.schema.json` | Format every question must follow |
| `schemas/misconception.schema.json` | Format every misconception must follow |

Skill codes everywhere: `py`, `sql`, `ds`.

## 1. Topics

Defined in `data/topics.json`. Every question, job and candidate key uses only these tags.

## 2. Misconceptions

- **Global per topic.** Questions point to misconceptions by ID; they never define their own.
- **ID format:** `<skill>.<topic without skill prefix>.<slug>`, for example `py.collections.tuple-mutable`, `sql.joins.left-join-drops-unmatched` or `ds.hash-tables.lookup-is-linear`.
- **How many:** 3–4 per topic, about 80–100 in total.
- **Coverage:** every misconception must be detectable by **at least 2 questions**, so it can be planted and confirmed.

Example:

```json
{
  "id": "sql.joins.left-join-drops-unmatched",
  "skill": "sql",
  "topic": "sql-joins",
  "description": "Believes LEFT JOIN drops left-table rows that have no match, like INNER JOIN",
  "example_wrong_answer": "Ben has no orders, so he won't appear in the result.",
  "source": "generated"
}
```

## 3. Jobs

Defined in `data/jobs.json`.

| Job | Min Python | Min SQL | Min DS |
| --- | --- | --- | --- |
| Junior Data Analyst | 2 | 2 | 1 |
| Junior Backend Engineer | 2 | 2 | 2 |
| Junior Data Engineer | 2 | 3 | 2 |

- **Still to write:** each job's `description`, a realistic 150–250-word job description.
- **Wording:** describe the work, not the tag names, so that topic extraction has to infer the topics.
- **Ground truth:** `required_topics` is the answer key for checking topic extraction.

## 4. Question format

Defined in `schemas/question.schema.json`.

- **ID:** `<SKILL>-L<level>-<number>`, for example `SQL-L2-014`.
- **`code`:** required for read-code questions.
- **`expected_output`:** for read-code questions whose code can be run. It holds the exact stdout (py/ds) or the result rows (sql). The validator runs the code and compares.
- **`tables`:** only for SQL questions that need data.
- **`status`:** stays `draft` until a teammate has checked the question, then becomes `checked`.

```json
{
  "id": "SQL-L2-014",
  "skill": "sql",
  "level": 2,
  "type": "read-code",
  "topics": ["sql-joins", "sql-null"],
  "question": "What does this query return?",
  "code": "SELECT c.name, o.id FROM customers c LEFT JOIN orders o ON c.id = o.customer_id;",
  "tables": [
    {"name": "customers", "columns": ["id", "name"], "rows": [[1, "Asha"], [2, "Ben"], [3, "Chen"]]},
    {"name": "orders", "columns": ["id", "customer_id"], "rows": [[10, 1], [11, 1], [12, 3]]}
  ],
  "model_answer": "Four rows: Asha with 10, Asha with 11, Ben with NULL, Chen with 12. LEFT JOIN keeps every customer; Ben has no order, so his order id is NULL.",
  "key_points": ["Every customer appears", "Asha appears twice", "Ben appears with a NULL order id"],
  "rubric": {
    "0": "Wrong rows, or Ben is missing",
    "1": "Ben is included but a row is missing or duplicated",
    "2": "All four rows correct, with the reason Ben appears"
  },
  "misconceptions": ["sql.joins.left-join-drops-unmatched"],
  "source": "generated",
  "status": "draft"
}
```

## 5. Levels

| Level | Python | SQL | Data structures |
| --- | --- | --- | --- |
| 1 Basics | Built-in types, control flow, simple functions | SELECT, WHERE, ORDER BY, DISTINCT, simple aggregates | Arrays, stacks, queues; what Big-O means |
| 2 Everyday use | Dicts and sets, comprehensions, classes, exceptions, mutability and aliasing, pandas basics | JOINs, GROUP BY / HAVING, NULL handling, subqueries | Hash tables, linked lists, BSTs, sorting and searching, choosing a structure |
| 3 Tricky cases and trade-offs | Generators and lazy evaluation, mutable default arguments, closures, GIL and concurrency, pandas pitfalls | Window functions, CTEs, indexes and query plans, transactions and isolation, `NOT IN` with NULL | Heaps, graphs (BFS/DFS, shortest path), amortised cost, worst cases (hash collisions, unbalanced trees) |

## 6. Size and split

- **Total:** 135 questions (3 skills × 3 levels × 15). The first 90 were chosen from 180 drafts; the revision plan (Change 1) grows the bank to 120–150, so on 2026-10-23 the best 45 kept-back drafts were added with `scripts/promote_pool.py` (5 per cell, preferring misconceptions on fewer than 3 questions; their `selection_record.json` entries have `"promoted": true`).
- **Each skill-level cell:** 9 explain and 6 read-code questions, for 81 explain and 54 read-code in total. The validator checks the intent rather than fixed numbers: every level of a skill has the same size (at least 10), and 60% of each cell is explain.
- **Coverage rules:**
    - Every topic appears in at least 2 questions.
    - Each skill-level cell covers at least 4 different topics.
    - Every job-required topic appears in at least 3 questions.

## 7. Rubric (0 / 1 / 2)

Each question lists 2–4 key points.

| Score | Meaning |
| --- | --- |
| 2 | Covers all key points, with no errors |
| 1 | Covers some key points, or has a minor error |
| 0 | Core idea wrong or missing, or shows one of the question's listed misconceptions |

## 8. Misconceptions per question

1–3, always chosen from `data/misconceptions.json`.

## 9. Language versions

- **Python:** 3.12, with pandas 3.x (copy-on-write is the default, so no SettingWithCopy questions). This matches the project `.venv`.
- **SQL:** standard SQL that behaves the same on SQLite.
- **Verification:** every SQL read-code answer is checked by running it with Python's built-in `sqlite3`.
    - Avoid SQLite quirks such as type affinity and bare columns in GROUP BY.
    - Index and transaction questions stay conceptual and are not executed.

## 10. SQL data in questions

- **Size:** at most 2 tables per question, 3–6 rows each.
- **NULLs:** include at least one NULL whenever the question touches NULLs, joins or aggregates.
- **Storage:** tables are stored as data in `tables`, so the same rows can be shown to the candidate and loaded into SQLite for checking.

## 11. Sources

- **Misconceptions:** adapted from published catalogs where they exist (Python, SQL), with the citation in `source`.
- **Everything else:** generated.
- **Proposal:** the Data section of the revised proposal says so.

## 12. Generator model

- **Generator:** a strong model (e.g. Claude Opus) generates questions and misconceptions.
- **Simulator vs grader:** the candidate simulator must come from a different model family than the grader.

## 13. What counts as "checked"

Before human review, a script checks the format against the schema and the coverage rules (section 6).

A question moves from `draft` to `checked` only when a teammate confirms all of these:

1. **Correct answer.** The model answer is correct. Read-code answers are verified by running the code (Python, or SQL in SQLite).
2. **Consistent marking.** The key points and rubric match the model answer.
3. **Relevant misconceptions.** They are specific and relevant to the question.
4. **Right level.** The level matches the definitions in section 5.
5. **Not a duplicate.** An automatic check flags pairs with embedding cosine similarity above 0.9.

## 14. Difficulty check

1. Two teammates each rate the same 21 random questions (7 per skill) on levels 1–3, without seeing the assigned level.
2. If a skill's assigned level disagrees with both raters on more than 2 of its 7 questions, re-review that skill's levels.
3. Report the agreement rate in the write-up.

## 15. Search index

- **Owner:** Part 1 builds the index, since it owns the bank. `scripts/build_index.py` embeds every question and saves the vectors to `index/`.
- **Search function:** Part 1 also provides a basic search function that filters by skill, level, topics and questions already asked.
- **Interview logic:** Part 2 decides what to ask, and when, using that function.
