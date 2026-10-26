"""OpenAI-compatible chat client implementing `shared.interfaces.LLMClient`.

Plain HTTP (httpx) against `{base_url}/chat/completions`, so it works with Gemini's
OpenAI-compatible endpoint, Groq, NVIDIA NIM, OpenRouter and similar.

Features: JSON-schema structured output, OpenAI-format tool calling (the assistant message
is returned untouched as `raw_message`, provider extras such as Gemini thought signatures
included), a disk cache, retries with backoff, and a minimum interval between calls.

Keys come from the environment / .env only, and are never logged or put in error messages.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[2]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Lowest `reasoning_effort` that gemini-flash-latest accepts through the OpenAI-compatible
# endpoint (see docs/llm.md). Used for grading, diagnosis and topic extraction.
DEFAULT_REASONING_EFFORT = "minimal"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Any failure of an LLM call (HTTP error, bad response)."""


class LLMOutputError(LLMError):
    """The model answered, but not with valid JSON / the requested schema / valid tool arguments."""


class LLMQuotaError(LLMError):
    """Rate limit that retrying will not fix soon (daily quota, or a very long retry delay)."""


# --------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------

def load_env(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Load KEY=VALUE lines from a .env file (default: the project's) into os.environ.

    Variables that are already set are NOT overridden. Returns the variables it set.
    Supports `#` comments, blank lines, `export KEY=...` and single/double quotes.
    A missing file is not an error.
    """
    p = Path(path) if path is not None else ROOT / ".env"
    loaded: dict[str, str] = {}
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError:
        return loaded
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def _error_body(resp: httpx.Response) -> Any:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:500]
    return body[0] if isinstance(body, list) and body else body


def _error_message(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("message", ""))[:500]
    return str(body)[:500]


def _retry_delay(resp: httpx.Response, body: Any) -> float | None:
    """Delay (seconds) the provider asked for: Retry-After header, Gemini's retryDelay, or 'retry in Ns'."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    err = body.get("error", body) if isinstance(body, dict) else {}
    for d in (err.get("details") or []) if isinstance(err, dict) else []:
        if isinstance(d, dict) and "retryDelay" in d:
            m = re.match(r"^([0-9.]+)s$", str(d["retryDelay"]))
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([0-9.]+)\s*s", _error_message(body), re.I)
    return float(m.group(1)) if m else None


class OpenAICompatClient:
    """`LLMClient` for any OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None,
                 cache: bool | None = None, cache_dir: str | os.PathLike | None = None,
                 min_interval_s: float | None = None, max_attempts: int | None = None,
                 max_wait_s: float | None = None, base_backoff_s: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self.cache_enabled = cache if cache is not None else os.environ.get("LLM_CACHE", "1") != "0"
        self.cache_dir = Path(cache_dir or os.environ.get("LLM_CACHE_DIR") or ROOT / ".cache" / "llm")
        self.min_interval_s = min_interval_s if min_interval_s is not None else _env_float("LLM_MIN_INTERVAL_S", 1.0)
        self.max_attempts = max_attempts if max_attempts is not None else int(_env_float("LLM_MAX_ATTEMPTS", 6))
        self.max_wait_s = max_wait_s if max_wait_s is not None else _env_float("LLM_MAX_WAIT_S", 90.0)
        self.base_backoff_s = base_backoff_s
        self._sleep, self._clock = sleep, clock
        self._lock = threading.Lock()
        self._last_call: float | None = None
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0}   # network requests actually sent

    def __repr__(self) -> str:
        return f"OpenAICompatClient(base_url={self.base_url!r})"

    # -- public API ---------------------------------------------------------

    def complete(self, messages: list[dict], *, model: str, temperature: float = 0.0,
                 json_schema: dict | None = None, max_tokens: int = 1024,
                 tools: list[dict] | None = None, reasoning_effort: str | None = None,
                 cache_tag: str | None = None) -> dict:
        key = self._cache_key(messages, model, temperature, json_schema, max_tokens, tools,
                              reasoning_effort, cache_tag)
        if self.cache_enabled:
            hit = self._cache_read(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                hit["cached"] = True
                return hit

        body: dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": temperature, "max_tokens": max_tokens}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "strict": True, "schema": json_schema}}
        if tools:
            body["tools"] = tools
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort

        data = self._post(body)
        result = self._parse(data, model, json_schema)
        result["cached"] = False
        if self.cache_enabled:
            self._cache_write(key, result)
        return result

    # -- cache --------------------------------------------------------------

    @staticmethod
    def _cache_key(*args: Any) -> str:
        blob = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> dict | None:
        try:
            return json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _cache_write(self, key: str, result: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_dir / f"{key}.tmp"
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.cache_dir / f"{key}.json")
        except OSError:
            pass   # a failed cache write must never fail the call

    # -- transport ----------------------------------------------------------

    def _throttle(self) -> None:
        with self._lock:
            if self._last_call is not None:
                wait = self.min_interval_s - (self._clock() - self._last_call)
                if wait > 0:
                    self._sleep(wait)
            self._last_call = self._clock()

    def _post(self, body: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        last = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            self.stats["calls"] += 1
            delay: float | None = None
            try:
                resp = self._http.post(url, headers=headers, json=body)
            except httpx.TransportError as e:   # network errors and timeouts
                last = f"network error: {type(e).__name__}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        raise LLMError("the endpoint returned a 200 response that is not JSON") from None
                err = _error_body(resp)
                msg = _error_message(err)
                last = f"HTTP {resp.status_code}: {msg}"
                if resp.status_code not in RETRY_STATUSES:
                    raise LLMError(f"LLM request failed ({last})")
                if resp.status_code == 429 and re.search(r"PerDay|per day|daily", json.dumps(err), re.I):
                    raise LLMQuotaError(f"daily quota exhausted ({last})")
                delay = _retry_delay(resp, err)
            if attempt == self.max_attempts:
                break
            wait = delay if delay is not None else self.base_backoff_s * 2 ** (attempt - 1)
            if delay is not None:
                wait += 0.5   # a little slack after the provider's own estimate
            if wait > self.max_wait_s:
                raise LLMQuotaError(f"rate limited, provider asks for {wait:.0f}s which exceeds "
                                    f"LLM_MAX_WAIT_S={self.max_wait_s:.0f} ({last})")
            self.stats["retries"] += 1
            self._sleep(wait)
        raise LLMError(f"LLM request failed after {self.max_attempts} attempts (last: {last})")

    # -- response -----------------------------------------------------------

    def _parse(self, data: dict, model: str, json_schema: dict | None) -> dict:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected response shape from the endpoint (no choices[0].message)") from None
        finish = choice.get("finish_reason")
        text = message.get("content") or ""
        if isinstance(text, list):   # content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))

        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = usage.get("total_tokens")
        reasoning = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        output = int(total) - prompt if total is not None and int(total) >= prompt else completion + reasoning

        result: dict[str, Any] = {"text": text, "model": data.get("model") or model,
                                  "input_tokens": prompt, "output_tokens": output,
                                  "raw_message": copy.deepcopy(message)}

        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
            except ValueError:
                raise LLMOutputError(f"tool call {fn.get('name')!r} has arguments that are not valid JSON") from None
            if not isinstance(args, dict):
                raise LLMOutputError(f"tool call {fn.get('name')!r} arguments are not a JSON object")
            calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name", ""), "arguments": args})
        if calls:
            result["tool_calls"] = calls

        if json_schema is not None and not calls:
            if finish == "length":
                raise LLMOutputError("the reply was cut off by max_tokens before the JSON was complete "
                                     "(hidden reasoning tokens count against max_tokens; raise it)")
            try:
                parsed = json.loads(_strip_fences(text))
            except ValueError as e:
                raise LLMOutputError(f"the model's reply is not valid JSON ({e}); got: {text[:200]!r}") from None
            _validate(parsed, json_schema)
            result["parsed"] = parsed
        return result


def _validate(parsed: Any, schema: dict) -> None:
    try:
        import jsonschema
    except ImportError:   # pragma: no cover - jsonschema is pinned in requirements
        return
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise LLMOutputError(f"the model's JSON does not match the schema at {path}: {e.message[:200]}") from None


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

_clients: dict[tuple[str, str], OpenAICompatClient] = {}


def get_client(role: str | None = None) -> OpenAICompatClient:
    """Client for `role`. Reads (after loading .env) LLM_BASE_URL_<ROLE> / LLM_API_KEY_<ROLE>
    first, then LLM_BASE_URL / LLM_API_KEY, then GEMINI_API_KEY on Gemini's endpoint. Clients
    with the same endpoint and key share one instance, so the minimum call interval holds
    across roles."""
    load_env()
    suffix = "_" + re.sub(r"[^A-Za-z0-9]+", "_", role).upper() if role else ""
    base = (os.environ.get(f"LLM_BASE_URL{suffix}") if suffix else None) or os.environ.get("LLM_BASE_URL")
    key = (os.environ.get(f"LLM_API_KEY{suffix}") if suffix else None) or os.environ.get("LLM_API_KEY")
    if not key and os.environ.get("GEMINI_API_KEY"):
        base, key = GEMINI_BASE_URL, os.environ["GEMINI_API_KEY"]
    if not base or not key:
        names = [f"LLM_BASE_URL{suffix}", f"LLM_API_KEY{suffix}"] if suffix else []
        raise RuntimeError("GEMINI_API_KEY, or LLM_BASE_URL and LLM_API_KEY, must be set (environment or .env)"
                           + (f"; per-role overrides: {', '.join(names)}" if names else ""))
    return client_for(base, key)


def client_for(base_url: str, api_key: str) -> OpenAICompatClient:
    """The shared client for this endpoint and key (one per pair, so the minimum call
    interval holds for everything that uses it)."""
    ident = (base_url.rstrip("/"), api_key)
    if ident not in _clients:
        _clients[ident] = OpenAICompatClient(base_url, api_key)
    return _clients[ident]
