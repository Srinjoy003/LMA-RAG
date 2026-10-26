"""The LLM client against a mocked HTTP transport: no network, no real key, no real sleeping."""

import json

import httpx
import pytest

from llm import client as client_mod
from llm.client import (LLMError, LLMOutputError, LLMQuotaError, OpenAICompatClient, get_client,
                        load_env)

SCHEMA = {"type": "object", "properties": {"score": {"type": "integer", "enum": [0, 1, 2]}},
          "required": ["score"], "additionalProperties": False}
MSGS = [{"role": "user", "content": "hi"}]


def ok(content='{"score": 2}', usage=None, message_extra=None, finish="stop", model="m-1"):
    message = {"role": "assistant", "content": content, **(message_extra or {})}
    return httpx.Response(200, json={
        "model": model, "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 25}})


class Server:
    """Plays a list of responses (the last repeats); records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        r = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    @property
    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


class Clock:
    """Fake time: sleeping advances it, so min-interval and backoff are testable instantly."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make(server, tmp_path, clock=None, **kw):
    clock = clock or Clock()
    defaults = dict(cache=True, cache_dir=tmp_path / "cache", min_interval_s=0.0, sleep=clock.sleep, clock=clock.time)
    defaults.update(kw)
    c = OpenAICompatClient("https://example.test/v1/", "sekret-key", transport=httpx.MockTransport(server), **defaults)
    return c, clock


# -- request / response -----------------------------------------------------------

def test_request_shape_and_structured_output(tmp_path):
    server = Server(ok())
    c, _ = make(server, tmp_path)
    r = c.complete(MSGS, model="m", temperature=0.3, json_schema=SCHEMA, max_tokens=500,
                   reasoning_effort="low")
    assert r["parsed"] == {"score": 2} and r["text"] == '{"score": 2}' and r["cached"] is False
    req, body = server.requests[0], server.bodies[0]
    assert str(req.url) == "https://example.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sekret-key"
    assert body["model"] == "m" and body["temperature"] == 0.3 and body["max_tokens"] == 500
    assert body["reasoning_effort"] == "low" and body["messages"] == MSGS
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == SCHEMA


def test_optional_fields_are_omitted_when_unset(tmp_path):
    server = Server(ok("plain"))
    c, _ = make(server, tmp_path)
    r = c.complete(MSGS, model="m")
    body = server.bodies[0]
    assert "response_format" not in body and "reasoning_effort" not in body and "tools" not in body
    assert r["text"] == "plain" and "parsed" not in r


def test_token_counts_include_reasoning_tokens(tmp_path):
    c, _ = make(Server(ok(usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 25})), tmp_path)
    r = c.complete(MSGS, model="m")
    assert r["input_tokens"] == 10 and r["output_tokens"] == 15   # total - prompt


def test_token_counts_without_total(tmp_path):
    usage = {"prompt_tokens": 7, "completion_tokens": 4,
             "completion_tokens_details": {"reasoning_tokens": 6}}
    c, _ = make(Server(ok(usage=usage)), tmp_path)
    r = c.complete(MSGS, model="m")
    assert r["input_tokens"] == 7 and r["output_tokens"] == 10


def test_invalid_json_raises_clear_error(tmp_path):
    c, _ = make(Server(ok("this is not json")), tmp_path)
    with pytest.raises(LLMOutputError, match="not valid JSON"):
        c.complete(MSGS, model="m", json_schema=SCHEMA)


def test_schema_violation_raises(tmp_path):
    c, _ = make(Server(ok('{"score": 5}')), tmp_path)
    with pytest.raises(LLMOutputError, match="does not match the schema"):
        c.complete(MSGS, model="m", json_schema=SCHEMA)


def test_truncated_reply_says_to_raise_max_tokens(tmp_path):
    c, _ = make(Server(ok('{"sco', finish="length")), tmp_path)
    with pytest.raises(LLMOutputError, match="max_tokens"):
        c.complete(MSGS, model="m", json_schema=SCHEMA)


def test_code_fenced_json_is_accepted(tmp_path):
    c, _ = make(Server(ok('```json\n{"score": 1}\n```')), tmp_path)
    assert c.complete(MSGS, model="m", json_schema=SCHEMA)["parsed"] == {"score": 1}


def test_failed_outputs_are_not_cached(tmp_path):
    server = Server(ok("garbage"), ok('{"score": 1}'))
    c, _ = make(server, tmp_path)
    with pytest.raises(LLMOutputError):
        c.complete(MSGS, model="m", json_schema=SCHEMA)
    assert c.complete(MSGS, model="m", json_schema=SCHEMA)["parsed"] == {"score": 1}


# -- tools ----------------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "get_q", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


def test_tool_calls_and_raw_message_round_trip(tmp_path):
    extra = {"content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_q", "arguments": '{"skill": "sql", "n": 2}'},
         "extra_content": {"google": {"thought_signature": "SIG=="}}}]}
    server = Server(httpx.Response(200, json={
        "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", **extra}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 9}}))
    c, _ = make(server, tmp_path)
    r = c.complete(MSGS, model="m", tools=TOOLS)
    assert server.bodies[0]["tools"] == TOOLS
    assert r["tool_calls"] == [{"id": "call_1", "name": "get_q", "arguments": {"skill": "sql", "n": 2}}]
    assert r["text"] == ""
    assert r["raw_message"] == {"role": "assistant", **extra}   # untouched, thought signature included
    assert r["raw_message"]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG=="


def test_tool_calls_survive_the_cache(tmp_path):
    msg = {"role": "assistant", "content": None, "extra_content": {"google": {"thought_signature": "S"}},
           "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    c, _ = make(Server(httpx.Response(200, json={"choices": [{"message": msg}]})), tmp_path)
    first = c.complete(MSGS, model="m", tools=TOOLS)
    second = c.complete(MSGS, model="m", tools=TOOLS)
    assert second["cached"] is True and second["tool_calls"] == first["tool_calls"]
    assert second["raw_message"] == msg


def test_bad_tool_arguments_raise(tmp_path):
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{oops"}}]}
    c, _ = make(Server(httpx.Response(200, json={"choices": [{"message": msg}]})), tmp_path)
    with pytest.raises(LLMOutputError, match="arguments"):
        c.complete(MSGS, model="m", tools=TOOLS)


# -- cache ----------------------------------------------------------------------

def test_cache_hit_skips_the_network_and_is_flagged(tmp_path):
    server = Server(ok())
    c, _ = make(server, tmp_path)
    first = c.complete(MSGS, model="m", json_schema=SCHEMA)
    second = c.complete(MSGS, model="m", json_schema=SCHEMA)
    assert first["cached"] is False and second["cached"] is True
    assert second["parsed"] == first["parsed"] and len(server.requests) == 1
    # a new client instance (new process) reads the same disk cache
    c2, _ = make(Server(ok("other")), tmp_path)
    assert c2.complete(MSGS, model="m", json_schema=SCHEMA)["cached"] is True
    assert list((tmp_path / "cache").glob("*.json"))


@pytest.mark.parametrize("change", [
    dict(model="other"), dict(temperature=0.5), dict(max_tokens=77), dict(reasoning_effort="low"),
    dict(cache_tag="vote-1"), dict(json_schema=SCHEMA), dict(tools=TOOLS), dict(messages=[{"role": "user", "content": "hello"}]),
])
def test_every_request_argument_is_part_of_the_cache_key(tmp_path, change):
    server = Server(ok("{}" if "json_schema" not in change else '{"score": 1}'))
    c, _ = make(server, tmp_path)
    base = dict(messages=MSGS, model="m")
    c.complete(**base)
    c.complete(**{**base, **change})
    assert len(server.requests) == 2


def test_cache_disabled_by_argument_and_by_env(tmp_path, monkeypatch):
    server = Server(ok())
    c, _ = make(server, tmp_path, cache=False)
    c.complete(MSGS, model="m")
    c.complete(MSGS, model="m")
    assert len(server.requests) == 2 and not (tmp_path / "cache").exists()

    monkeypatch.setenv("LLM_CACHE", "0")
    server = Server(ok())
    c = OpenAICompatClient("https://x/v1", "k", transport=httpx.MockTransport(server), cache_dir=tmp_path / "c2",
                           min_interval_s=0)
    c.complete(MSGS, model="m")
    c.complete(MSGS, model="m")
    assert len(server.requests) == 2 and c.cache_enabled is False


# -- retries --------------------------------------------------------------------

def err(status, message="boom", details=None, headers=None):
    return httpx.Response(status, json={"error": {"code": status, "message": message, "details": details or []}},
                          headers=headers or {})


def test_retries_on_5xx_with_exponential_backoff(tmp_path):
    server = Server(err(503), err(500), err(502), ok())
    c, clock = make(server, tmp_path, base_backoff_s=2.0)
    assert c.complete(MSGS, model="m")["text"] == '{"score": 2}'
    assert len(server.requests) == 4 and clock.sleeps == [2.0, 4.0, 8.0]
    assert c.stats["retries"] == 3


def test_429_honours_gemini_retry_delay(tmp_path):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]
    server = Server(err(429, "slow down", details), ok())
    c, clock = make(server, tmp_path)
    c.complete(MSGS, model="m")
    assert 12 <= clock.sleeps[0] <= 13.5


def test_429_honours_retry_after_header_and_message(tmp_path):
    c, clock = make(Server(err(429, headers={"Retry-After": "7"}), ok()), tmp_path)
    c.complete(MSGS, model="m")
    assert 7 <= clock.sleeps[0] <= 8.5
    c, clock = make(Server(err(429, "Please retry in 3.5s."), ok()), tmp_path, cache=False)
    c.complete(MSGS, model="m")
    assert 3.5 <= clock.sleeps[0] <= 5


def test_gemini_list_shaped_error_body(tmp_path):
    body = [{"error": {"code": 429, "message": "x", "details": [{"retryDelay": "2s"}]}}]
    c, clock = make(Server(httpx.Response(429, json=body), ok()), tmp_path)
    c.complete(MSGS, model="m")
    assert 2 <= clock.sleeps[0] <= 3


def test_gives_up_after_max_attempts(tmp_path):
    server = Server(err(503, "overloaded"))
    c, clock = make(server, tmp_path, max_attempts=3)
    with pytest.raises(LLMError, match="after 3 attempts.*overloaded"):
        c.complete(MSGS, model="m")
    assert len(server.requests) == 3 and len(clock.sleeps) == 2


def test_client_errors_are_not_retried(tmp_path):
    server = Server(err(400, "Request contains an invalid argument."))
    c, clock = make(server, tmp_path)
    with pytest.raises(LLMError, match="HTTP 400"):
        c.complete(MSGS, model="m")
    assert len(server.requests) == 1 and clock.sleeps == []


def test_daily_quota_fails_fast(tmp_path):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
               {"retryDelay": "66548s"}]
    server = Server(err(429, "You exceeded your current quota", details))
    c, clock = make(server, tmp_path)
    with pytest.raises(LLMQuotaError, match="daily quota"):
        c.complete(MSGS, model="m")
    assert len(server.requests) == 1 and clock.sleeps == []


def test_very_long_retry_delay_is_not_waited_out(tmp_path):
    details = [{"retryDelay": "600s"}]
    server = Server(err(429, "per minute", details))
    c, clock = make(server, tmp_path, max_wait_s=90)
    with pytest.raises(LLMQuotaError, match="LLM_MAX_WAIT_S"):
        c.complete(MSGS, model="m")
    assert clock.sleeps == []


def test_network_errors_are_retried(tmp_path):
    server = Server(httpx.ConnectError("no route"), httpx.ReadTimeout("slow"), ok())
    c, clock = make(server, tmp_path)
    assert c.complete(MSGS, model="m")["text"]
    assert len(server.requests) == 3 and len(clock.sleeps) == 2


def test_api_key_never_appears_in_errors_or_repr(tmp_path):
    c, _ = make(Server(err(401, "bad key")), tmp_path)
    with pytest.raises(LLMError) as e:
        c.complete(MSGS, model="m")
    assert "sekret-key" not in str(e.value) and "sekret-key" not in repr(c)


def test_minimum_interval_between_calls(tmp_path):
    server = Server(ok())
    c, clock = make(server, tmp_path, min_interval_s=1.0, cache=False)
    c.complete(MSGS, model="m")
    c.complete(MSGS, model="m")
    assert clock.sleeps == [1.0]
    clock.now += 5       # already waited long enough
    c.complete(MSGS, model="m")
    assert clock.sleeps == [1.0]


def test_cache_hits_do_not_wait(tmp_path):
    c, clock = make(Server(ok()), tmp_path, min_interval_s=1.0)
    c.complete(MSGS, model="m")
    c.complete(MSGS, model="m")
    assert clock.sleeps == []


def test_min_interval_comes_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "2.5")
    c = OpenAICompatClient("https://x/v1", "k", transport=httpx.MockTransport(Server(ok())), cache_dir=tmp_path)
    assert c.min_interval_s == 2.5


# -- load_env / get_client ---------------------------------------------------------

def test_load_env_parses_and_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\n\nA_ONE=1\nexport A_TWO = two words\nA_QUOTED=\"quoted # value\"\n"
                 "A_SINGLE='s'\nA_EXISTING=from_file\nA_TRAIL=val # note\nnot a line\nA_EQ=a=b\n", encoding="utf-8")
    for k in ["A_ONE", "A_TWO", "A_QUOTED", "A_SINGLE", "A_TRAIL", "A_EQ"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("A_EXISTING", "from_env")
    loaded = load_env(f)
    import os
    assert os.environ["A_ONE"] == "1" and os.environ["A_TWO"] == "two words"
    assert os.environ["A_QUOTED"] == "quoted # value" and os.environ["A_SINGLE"] == "s"
    assert os.environ["A_TRAIL"] == "val" and os.environ["A_EQ"] == "a=b"
    assert os.environ["A_EXISTING"] == "from_env" and "A_EXISTING" not in loaded
    assert set(loaded) == {"A_ONE", "A_TWO", "A_QUOTED", "A_SINGLE", "A_TRAIL", "A_EQ"}
    for k in loaded:
        monkeypatch.delenv(k, raising=False)   # monkeypatch did not set these; clean up


def test_load_env_missing_file_is_fine(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_get_client_role_overrides_and_fallback(monkeypatch):
    monkeypatch.setattr(client_mod, "load_env", lambda *a, **k: {})
    monkeypatch.setattr(client_mod, "_clients", {})
    monkeypatch.setenv("LLM_BASE_URL", "https://default.test/v1/")
    monkeypatch.setenv("LLM_API_KEY", "default-key")
    monkeypatch.setenv("LLM_BASE_URL_SIMULATOR", "https://sim.test/v1")
    monkeypatch.setenv("LLM_API_KEY_SIMULATOR", "sim-key")
    default, sim = get_client(), get_client("simulator")
    assert default.base_url == "https://default.test/v1" and default._api_key == "default-key"
    assert sim.base_url == "https://sim.test/v1" and sim._api_key == "sim-key"
    grader = get_client("grader")           # no override: falls back to the default endpoint
    assert grader is default and get_client() is default


def test_get_client_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(client_mod, "load_env", lambda *a, **k: {})
    monkeypatch.setattr(client_mod, "_clients", {})
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_client()


def test_get_client_falls_back_to_the_gemini_key(monkeypatch):
    monkeypatch.setattr(client_mod, "load_env", lambda *a, **k: {})
    monkeypatch.setattr(client_mod, "_clients", {})
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    client = get_client()
    assert client._api_key == "gemini-key" and "generativelanguage.googleapis.com" in client.base_url
