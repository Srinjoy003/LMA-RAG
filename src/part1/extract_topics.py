"""Extract interview topics from a job description with one LLM call.

The model's answer is constrained to the topic ids in data/topics.json by a JSON
schema whose items are an enum of those ids; anything else is dropped anyway.

The call goes through the provider-agnostic `LLMClient` (src/llm); endpoint and key come
from GEMINI_API_KEY, or LLM_BASE_URL / LLM_API_KEY (environment or .env).
"""

from __future__ import annotations

import os

from llm.client import DEFAULT_REASONING_EFFORT, get_client
from shared.interfaces import LLMClient, ModelSettings

from .bank import load_bank

DEFAULT_MODEL = ModelSettings().topic_extractor


def get_model() -> str:
    return os.environ.get("TOPIC_MODEL") or DEFAULT_MODEL


def _topic_catalogue(topics: dict) -> list[tuple[str, str]]:
    return [(t["id"], t["name"]) for items in topics.values() for t in items]


def build_schema(topic_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "topics": {"type": "array", "items": {"type": "string", "enum": topic_ids}},
        },
        "required": ["topics"],
        "additionalProperties": False,
    }


def build_prompt(job_description: str, catalogue: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (system prompt, user message)."""
    topic_lines = "\n".join(f"- {tid}: {name}" for tid, name in catalogue)
    system = (
        "You help prepare technical interviews for junior roles. Given a job description, "
        "decide which of the topics below a candidate for this job must know. The description "
        "talks about the work, not about topic names, so infer the topics from the tasks "
        "described. Choose a topic only if the work clearly needs it; do not list everything. "
        "Answer only with topic ids from this list.\n\nTopics:\n" + topic_lines
    )
    user = f"Job description:\n\n{job_description}"
    return system, user


def extract_topics(job_description: str, llm: LLMClient | None = None, model: str | None = None,
                   topics: dict | None = None,
                   reasoning_effort: str | None = DEFAULT_REASONING_EFFORT) -> list[str]:
    """Return the topic ids (from topics.json only) that the job description needs.

    `llm` is any `LLMClient` (pass a fake in tests); by default `llm.get_client()`.
    `model` defaults to TOPIC_MODEL or `ModelSettings().topic_extractor`.
    """
    if not job_description or not job_description.strip():
        return []
    topics = topics if topics is not None else load_bank().topics
    catalogue = _topic_catalogue(topics)
    topic_ids = [tid for tid, _ in catalogue]
    system, user = build_prompt(job_description, catalogue)

    if llm is None:
        llm = get_client("topic_extractor")
    response = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model or get_model(), temperature=0.0, json_schema=build_schema(topic_ids),
        max_tokens=4096, reasoning_effort=reasoning_effort, cache_tag="topics")

    parsed = response.get("parsed")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("topics"), list):
        raise RuntimeError("could not parse the model's answer as the expected JSON: "
                           f"{(response.get('text') or '')[:200]!r}")
    valid = set(topic_ids)
    result: list[str] = []
    for t in parsed["topics"]:
        if t in valid and t not in result:
            result.append(t)
    return result


class LLMTopicExtractor:
    """`shared.interfaces.TopicExtractor`: job description -> topic ids, via an LLM."""

    def __init__(self, llm: LLMClient | None = None, model: str | None = None, topics: dict | None = None,
                 reasoning_effort: str | None = DEFAULT_REASONING_EFFORT):
        self.llm, self.model, self.topics, self.reasoning_effort = llm, model, topics, reasoning_effort

    def __call__(self, job_description: str) -> list[str]:
        return extract_topics(job_description, llm=self.llm, model=self.model, topics=self.topics,
                              reasoning_effort=self.reasoning_effort)


def score_topics(predicted: list[str], expected: list[str]) -> dict:
    """Precision, recall and F1 of predicted topics against the answer key."""
    p, e = set(predicted), set(expected)
    tp = len(p & e)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(e) if e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "missed": sorted(e - p), "extra": sorted(p - e),
    }
