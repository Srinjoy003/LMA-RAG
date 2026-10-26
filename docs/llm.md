# LLM layer: how to use it

Code: `src/llm/client.py` (client), `src/part3/grader.py` (`PromptedGrader`), `src/part3/diagnoser.py` (`StructuredDiagnoser`), `src/part1/extract_topics.py` (`extract_topics`, `LLMTopicExtractor`). Tests: `tests/test_llm_client.py`, `test_grader.py`, `test_diagnoser.py`, `test_extract_topics.py` (no network; they use `tests/fake_llm.py` and a mocked HTTP transport).

## Setup

Put your Gemini key in `.env` in the project root (git-ignored) or in the environment. The environment wins; `.env` never overrides a variable that is already set.

```
GEMINI_API_KEY=...
GROQ_API_KEY=...        # optional: Groq
```

The older pair `LLM_API_KEY` / `LLM_BASE_URL` still works and, when set, is the default endpoint.

```python
from llm import get_client
from part3 import PromptedGrader, StructuredDiagnoser
llm = get_client()                      # or get_client("simulator")
grader = PromptedGrader(llm, "gemini-3.5-flash-lite")
diagnoser = StructuredDiagnoser(llm, "gemini-3.5-flash-lite", misconceptions_by_id)
```

The key is only read from the environment, sent in the `Authorization` header and never logged or put in error messages.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | | Gemini key; the default endpoint when `LLM_API_KEY` is not set |
| `LLM_BASE_URL`, `LLM_API_KEY` | `GEMINI_API_KEY` on Gemini's endpoint | Default endpoint (any OpenAI-compatible `/chat/completions`) |
| `LLM_BASE_URL_<ROLE>`, `LLM_API_KEY_<ROLE>` | falls back to the defaults | Per-role endpoint, e.g. `LLM_BASE_URL_SIMULATOR` for `get_client("simulator")`. Role names are upper-cased, other characters become `_`. Topic extraction asks for the role `topic_extractor`. |
| `LLM_CACHE` | `1` | `0` turns the disk cache off |
| `LLM_CACHE_DIR` | `.cache/llm` | Cache folder |
| `LLM_MIN_INTERVAL_S` | `1.0` | Minimum seconds between network calls (cache hits do not wait) |
| `LLM_MAX_ATTEMPTS` | `6` | Attempts per call (retries on 429, 500, 502, 503, 504 and network errors) |
| `LLM_MAX_WAIT_S` | `90` | If the provider asks for a longer wait than this, fail with `LLMQuotaError` instead of sleeping |
| `TOPIC_MODEL` | `ModelSettings().topic_extractor` | Model for topic extraction |

## Behaviour of the client

- **Cache.** Every response is stored in `.cache/llm/<sha256>.json`, keyed by messages, model, temperature, schema, max_tokens, tools, reasoning_effort and `cache_tag`. A hit returns the stored response with `cached: True` and costs nothing. Failed or invalid replies are never cached. The grader and diagnoser give each vote its own tag (`vote-0`, `vote-1`, ...), so votes at temperature 0.7 are different calls but a re-run replays them. Delete the folder (or set `LLM_CACHE=0`) to get fresh samples.
- **Retries.** Exponential backoff (2, 4, 8 ... s). A delay given by the provider (`Retry-After`, Gemini's `retryDelay`, or "retry in Ns") is used instead. A per-day quota error (`...PerDay...`) raises `LLMQuotaError` straight away, because waiting will not help.
- **Structured output.** `json_schema` is sent as `response_format` (strict). The reply is parsed and validated against the schema; otherwise `LLMOutputError` explains what was wrong. Gemini's hidden thinking tokens count against `max_tokens`, so grader/diagnoser use 2048; a reply cut off by `max_tokens` raises an error that says so.
- **Tools.** `tool_calls` come back as `{id, name, arguments: dict}`. `raw_message` is the assistant message exactly as returned (Gemini's `extra_content.google.thought_signature` included) and survives the cache; send it back unchanged in the next request.
- **Tokens.** `output_tokens` is `total_tokens - prompt_tokens` (so it includes reasoning tokens), or `completion + reasoning` if the provider gives no total.
- **Graders and diagnosers** drop a vote whose output is unusable and carry on with the rest (raise only if none are usable). Network and quota errors propagate.

## Reasoning effort

Default for grading, diagnosis and topic extraction: `DEFAULT_REASONING_EFFORT = "minimal"` (in `llm/client.py`). Tested through the Gemini OpenAI-compatible endpoint (with a JSON schema that includes a nullable enum):

| value | `gemini-flash-lite-latest` |
| --- | --- |
| `none` | HTTP 400 "invalid argument" |
| `minimal` | works (74 output tokens on a small diagnosis) |
| `low` | works (57 output tokens) |

`gemini-flash-latest` could NOT be tested: when this was written it already answered 429 with a per-day quota of 20 requests (see below). `minimal` is Gemini 3's lowest thinking level, so it should work there too; if it returns a 400, change the constant to `low`. `medium`/`high` were not tried. Pass `reasoning_effort=` to the grader/diagnoser to override it; `None` omits the field.

## Free-tier notes (observed)

- `gemini-flash-latest` currently resolves to `gemini-3.8-flash`, whose free quota is **20 requests per day per project**. It was already used up (retry in about 18 hours), so every call returned 429 `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. Quotas are per model.
- `gemini-flash-lite-latest` worked for 60+ calls in one session without hitting a limit, with a 1 s minimum interval. The 54-call smoke test needed 3 automatic retries (transient errors, handled by the client).
- Use the cache: a full smoke run is 54 calls and a topic evaluation 3; repeats are free.
- `GET {base}/models` lists the models (it does not count as a generation request).

## Smoke test and evaluation

```
.venv\Scripts\python.exe scripts\smoke_llm.py --model gemini-flash-lite-latest
.venvScriptspython.exe scriptssmoke_llm.py                  # default model: gemini-3.5-flash-lite
TOPIC_MODEL=gemini-flash-lite-latest .venv\Scripts\python.exe scripts\eval_topic_extraction.py
```

(PowerShell: `$env:TOPIC_MODEL = "gemini-flash-lite-latest"` first.)

`smoke_llm.py` takes 6 level-2 questions (an "explain" and a "read-code" per skill), grades the model answer and a wrong answer made from a misconception's `example_wrong_answer`, diagnoses the wrong answer, and prints a summary. Options: `--votes`, `--effort`, `--level`. It is not part of pytest. If the daily quota runs out it stops and summarises what it has; run it again later and the cached part is free.

`eval_topic_extraction.py` prints precision, recall and F1 per job in `data/jobs.json`, with the missed and extra topics.

## Notes for the engine

- `Grade.votes` lists only usable votes; `agreement` is measured over them.
- A 3-way tie (or any tie among the top marks) takes the lower median of all marks, e.g. votes 0, 1, 2 give 1.
- The diagnoser sets `misconception_id` only when the cause is `misconception` and the id is on the question's list; otherwise `None`. `follow_up` is taken from the first winning vote that has one (it may be set even for `gap`; the engine already ignores it then). Each vote also carries `confidence_expressed`.
- `PromptedGrader` loads misconception descriptions from `data/misconceptions/*.json` unless `misconceptions=` is passed, because the `Grader` protocol does not hand them over.
- Candidate text is wrapped in `<<< >>>` and the prompts tell the model to treat it as data, to resist the safety attacks.

## Default model (2026-10-19)

All roles default to **`gemini-3.5-flash-lite`**, pinned by exact name: `-latest` aliases silently change the model behind them, which would break reproducibility. `gemini-flash-latest` (= `gemini-3.8-flash`) has only 20 free requests per day; `gemini-2.5-flash` and `gemini-2.5-flash-lite` are retired for new users; `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` answered normally.

## Switching provider (Gemini, Groq or Ollama)

`src/llm/providers.py` names the providers the app can switch between. Each sets the endpoint, key, model and reasoning effort for every role.

| Provider | Variables | Model | `reasoning_effort` |
| --- | --- | --- | --- |
| `gemini` | `GEMINI_API_KEY` (or the older `LLM_API_KEY`), optional `GEMINI_BASE_URL` | the config's (`gemini-3.5-flash-lite`) | `minimal` |
| `groq` | `GROQ_API_KEY`, optional `GROQ_BASE_URL` (default `https://api.groq.com/openai/v1`) and `GROQ_MODEL` | `openai/gpt-oss-20b` | `low` (gpt-oss rejects `minimal`) |
| `ollama` | none needed; optional `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) and `OLLAMA_MODEL` | `llama3` | not sent |

- **App:** choose Gemini, Groq or Ollama under Mode in the sidebar. With Groq, the saved result's `config.models` names the Groq model.
- **Smoke test:** `scripts/smoke_llm.py --provider groq`.
- **Code:** `apply_provider(config, name)` sets the model names, then `build_components(config, live=True, provider=name)`.
- **Caveat:** the experiments require the simulator (default `gpt-oss-120b`) to come from a different family than the grader, so don't grade with gpt-oss in the experiments.
