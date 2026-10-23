"""Builds the `Components` for an interview from a config (used by the app and scripts).

    components = build_components(config, live=False)    # stubs, no API key
    components = build_components(config, live=True)     # prompted grader + diagnoser (needs a key)

Live imports (llm, part3, part1.extract_topics) happen inside the function, so the stub path
never needs them. If a live module is missing or fails to import, `FactoryError` says which.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

from shared.interfaces import Components

ROOT = Path(__file__).resolve().parents[2]

# `ModelSettings.grader` names that are placeholders (a fine-tuned grader that does not exist yet)
PLACEHOLDER_MODELS = frozenset({"finetuned-grader"})
LIVE_GRADER_FALLBACK = "gemini-3.5-flash-lite"


class FactoryError(RuntimeError):
    """Live components could not be built."""


def _zero() -> dict:
    return {"llm_calls": 0, "api_calls": 0, "input_tokens": 0, "output_tokens": 0}


class MeteredLLM:
    """Wraps an LLM client and counts every call made through it (grader, diagnoser,
    topic extractor and agent all share one), so a result can report its true usage.
    `api_calls` excludes cache hits, i.e. what actually used quota."""

    def __init__(self, inner):
        self.inner = inner
        self._totals = {**_zero(), "by_model": {}}

    def complete(self, messages, **kwargs):
        response = self.inner.complete(messages, **kwargs)
        model = response.get("model") or kwargs.get("model") or "unknown"
        for target in (self._totals, self._totals["by_model"].setdefault(model, _zero())):
            target["llm_calls"] += 1
            target["api_calls"] += 0 if response.get("cached") else 1
            target["input_tokens"] += response.get("input_tokens", 0) or 0
            target["output_tokens"] += response.get("output_tokens", 0) or 0
        return response

    def usage_totals(self) -> dict:
        return {**{k: v for k, v in self._totals.items() if k != "by_model"},
                "by_model": {m: dict(v) for m, v in self._totals["by_model"].items()}}


def usage_delta(before: dict, after: dict) -> dict:
    """Usage between two `MeteredLLM.usage_totals()` snapshots."""
    keys = _zero().keys()
    out = {k: after[k] - before[k] for k in keys}
    out["by_model"] = {}
    for model, counts in after["by_model"].items():
        prev = before["by_model"].get(model, _zero())
        delta = {k: counts[k] - prev[k] for k in keys}
        if delta["llm_calls"]:
            out["by_model"][model] = delta
    return out


def grader_model(config) -> str:
    """The grader model for live use: the configured one if it is a real name, else the fallback."""
    name = config.models.grader
    return LIVE_GRADER_FALLBACK if (not name or name in PLACEHOLDER_MODELS) else name


def _import(module: str):
    try:
        return importlib.import_module(module)
    except Exception as e:      # missing module, or it fails while importing
        raise FactoryError(f"live mode needs the module {module!r}, which could not be imported "
                           f"({type(e).__name__}: {e})") from e


def _question_source(bank, root: Path):
    """The saved search index if there is one. Otherwise an index with a trivial embedding:
    the engine only filters by skill and level, so it never needs real embeddings."""
    from part1.index import QuestionIndex
    index_dir = root / "index"
    if (index_dir / "meta.json").exists():
        return QuestionIndex.load(index_dir, bank=bank)
    import numpy as np
    return QuestionIndex.build(bank, embed_fn=lambda texts: np.ones((len(texts), 1), dtype=np.float32))


def live_status(provider: str | None = None) -> tuple[bool, str]:
    """(ready, message): is an LLM endpoint and key configured? Never raises.
    `provider` is a name from `llm.providers.PROVIDERS`; None checks LLM_BASE_URL / LLM_API_KEY."""
    if provider is not None:
        try:
            return importlib.import_module("llm.providers").provider_status(provider)
        except Exception as e:
            return False, f"The LLM providers could not be loaded ({type(e).__name__}: {e})."
    try:
        client_module = importlib.import_module("llm.client")
        client_module.load_env()
    except Exception as e:
        return False, f"The LLM client could not be loaded ({type(e).__name__}: {e})."
    import os
    if os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"):
        return True, "LLM endpoint and key found."
    return False, ("No API key found. Set LLM_BASE_URL and LLM_API_KEY in the environment or in the "
                   "project's .env file, or switch to the offline stub mode.")


def build_components(config, *, live: bool, root: str | Path | None = None,
                     provider: str | None = None) -> Components:
    """`provider` (live only) picks the endpoint, key and reasoning effort from
    `llm.providers`; pass the config through `apply_provider` first so the model names match.
    None uses LLM_BASE_URL / LLM_API_KEY with the components' default reasoning effort."""
    root = Path(root) if root else ROOT
    from part1.bank import load_bank
    bank = load_bank(root)
    source = _question_source(bank, root)
    records = bank.misconception_by_id

    if not live:
        from shared.stubs import TagDiagnoser, TagGrader
        return Components(question_source=source, grader=TagGrader(votes=config.grader_votes),
                          diagnoser=TagDiagnoser(), misconceptions=records)

    client_module = _import("llm.client")
    grader_module = _import("part3.grader")
    diagnoser_module = _import("part3.diagnoser")
    effort = {}   # extra keyword for the grader, diagnoser and topic extractor
    try:
        if provider is None:
            llm = MeteredLLM(client_module.get_client())
        else:
            providers = _import("llm.providers")
            llm = MeteredLLM(providers.provider_client(provider))
            effort = {"reasoning_effort": providers.PROVIDERS[provider].reasoning_effort}
    except FactoryError:
        raise
    except Exception as e:
        raise FactoryError(f"live mode needs an LLM endpoint and key: {e}") from e

    grader = grader_module.PromptedGrader(llm, model=grader_model(config), votes=config.grader_votes, **effort)
    diagnoser = diagnoser_module.StructuredDiagnoser(
        llm, model=config.models.diagnoser, misconceptions=records, votes=config.diagnosis_votes, **effort)

    extractor = None
    if config.use_topic_extraction:
        topics_module = _import("part1.extract_topics")
        extractor_class = getattr(topics_module, "LLMTopicExtractor", None)
        if extractor_class is None:
            warnings.warn("part1.extract_topics has no LLMTopicExtractor yet; the job's required_topics "
                          "from jobs.json are used instead", stacklevel=2)
        else:
            extractor = extractor_class(llm, config.models.topic_extractor, **effort)

    return Components(question_source=source, grader=grader, diagnoser=diagnoser, topic_extractor=extractor,
                      llm=llm, misconceptions=records)
