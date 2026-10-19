"""A scripted `LLMClient` for tests: no network, records every call, enforces the JSON schema."""

import json

import jsonschema

from llm.client import LLMOutputError


class FakeLLM:
    """`replies` is a list of replies used in order (the last one repeats), or a function
    `(call_index, kwargs) -> reply`. A reply is a dict (the parsed JSON), a string (raw text),
    or an Exception (raised)."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, messages, *, model, temperature=0.0, json_schema=None, max_tokens=1024,
                 tools=None, reasoning_effort=None, cache_tag=None):
        kwargs = dict(messages=messages, model=model, temperature=temperature, json_schema=json_schema,
                      max_tokens=max_tokens, tools=tools, reasoning_effort=reasoning_effort, cache_tag=cache_tag)
        i = len(self.calls)
        self.calls.append(kwargs)
        reply = self.replies(i, kwargs) if callable(self.replies) else self.replies[min(i, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        out = {"text": text, "model": model, "input_tokens": 10, "output_tokens": 5, "cached": False}
        if json_schema is not None:
            try:
                parsed = json.loads(text)
                jsonschema.validate(parsed, json_schema)
            except (ValueError, jsonschema.ValidationError) as e:
                raise LLMOutputError(f"fake LLM: reply does not match the schema: {e}") from None
            out["parsed"] = parsed
        return out
