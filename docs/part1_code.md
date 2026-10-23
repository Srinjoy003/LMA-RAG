# Part 1 code: bank loader, validator, search index, topic extraction

All commands run from the project root with the project virtual env (Python 3.12).

## Install

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pulls the CPU-only build of torch. The first index build downloads the
`all-MiniLM-L6-v2` model (about 90 MB) from Hugging Face.

## Layout

| Path | What it does |
| --- | --- |
| `src/part1/bank.py` | `load_bank()` loads topics, jobs, questions, misconceptions. Missing files are empty. Lookups: `get_question`, `questions_for(skill, level)`, `questions_by_topic`, `misconceptions_for`, `get_job`. |
| `src/part1/validate.py` | The checks behind `validate_bank.py`. |
| `src/part1/execute.py` | Runs read-code questions (Python subprocess, SQLite in memory). |
| `src/part1/index.py` | `QuestionIndex`: build, save, load, search, near duplicates. |
| `src/part1/extract_topics.py` | `extract_topics(job_description)`. |
| `src/part1/pool.py` | Picks the best kept-back drafts from `data/pool/` to grow the bank (behind `promote_pool.py`). |
| `scripts/` | `validate_bank.py`, `build_index.py`, `eval_topic_extraction.py`, `promote_pool.py`. |
| `tests/` | pytest suite; `tests/fixtures/make_bank.py` builds a small valid bank for the tests. |

## Validate the bank

```
.venv\Scripts\python.exe scripts\validate_bank.py
```

Options: `--skill py|sql|ds` (one skill), `--no-exec` (do not run read-code questions),
`--dupes` (warn about question pairs with embedding cosine similarity above 0.9; loads the model).
Exit code 1 if there is any error; warnings do not change the exit code.

Errors: schema violations, duplicate IDs, ID prefix or level not matching `skill` / `level`,
misconception ID not matching its skill and topic, unknown topics or misconceptions (or ones from
another skill), unknown job topics, the coverage rules of spec section 6 (per skill that has data),
and read-code output that differs from `expected_output`.
Warnings: empty job description, read-code question without `expected_output`, a skill with no data.

How execution is compared:

- py / ds: the code runs with the same interpreter (10 s timeout). Stdout is compared after
  stripping trailing whitespace on each line and trailing newlines.
- sql: tables are created without column types and filled (`null` becomes NULL). All statements
  run; the last one that returns rows is compared. If that statement has a top-level `ORDER BY`
  the rows must match in order, otherwise as a multiset. `null` equals NULL and `2` equals `2.0`.

## Build the search index

```
.venv\Scripts\python.exe scripts\build_index.py
```

Writes `index/embeddings.npy` (one unit vector per question) and `index/meta.json` (question IDs
in row order, model name, hash of the embedded text). Text embedded per question: question text,
code, topic names, key points. Re-run it after the questions change; `QuestionIndex.load` warns
when the hash no longer matches the bank, and raises if indexed IDs are missing from the bank.

## Search from Python

```python
import sys; sys.path.insert(0, "src")
from part1.index import QuestionIndex

index = QuestionIndex.load()          # reads index/ and data/
results = index.search(
    query="mutable default argument",     # optional; cosine similarity to the question text
    skill="py",
    levels=[2, 3],
    prefer_topics=["py-functions-scope"], # more overlap ranks first
    exclude_ids={"PY-L2-001"},            # already asked
    k=5,
)
for question, score in results:
    print(question["id"], round(score, 3), question["question"])

index.near_duplicates(threshold=0.9)      # [(id_a, id_b, cosine), ...]
```

Filtering by skill, levels and `exclude_ids` happens first. Ranking is by topic overlap with
`prefer_topics`, then by cosine similarity to `query`, then by ID. The score is the cosine
similarity when a query is given, otherwise the overlap count.

The embedding function is injectable: `QuestionIndex.build(bank, embed_fn=f)` and
`QuestionIndex.load(embed_fn=f)` accept any `f(list[str]) -> array of shape (n, d)`. The tests use
a small deterministic fake, so they never download a model.

## Topic extraction

```python
from part1.extract_topics import extract_topics
topics = extract_topics(job_description)   # list of topic ids from data/topics.json
```

One call to the Anthropic API. The answer is constrained by a JSON schema whose items are an enum of
the 27 topic ids; anything else is dropped. Default model `claude-sonnet-5-5`; set the environment
variable `TOPIC_MODEL` to use another one. The key is read only from `ANTHROPIC_API_KEY`.

Evaluate against the answer key in `data/jobs.json`:

```
$env:ANTHROPIC_API_KEY = "..."        # PowerShell
.venv\Scripts\python.exe scripts\eval_topic_extraction.py
```

Prints precision, recall and F1 per job with the missed and extra topics, and the average. Jobs with
an empty description are skipped. Without an API key the script exits with a message.

## Tests

```
.venv\Scripts\python.exe -m pytest
```

No network and no model download: the Anthropic client is mocked and the index tests use the fake
embedder.
