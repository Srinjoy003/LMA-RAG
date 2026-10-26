"""Named LLM providers the app can switch between: endpoint, key, model and reasoning effort.

    gemini  GEMINI_API_KEY (the older LLM_API_KEY also works), optional GEMINI_BASE_URL;
            keeps the config's model names
    groq    GROQ_API_KEY, optional GROQ_BASE_URL and GROQ_MODEL (default openai/gpt-oss-20b)
    ollama  a local Ollama server, no key needed; optional OLLAMA_BASE_URL and OLLAMA_MODEL
            (default llama3)

Keys come from the environment / .env only, like everywhere else. Extra accounts' keys can sit
beside the usual ones with a suffix (GROQ_API_KEY_2, GEMINI_API_KEY_2) and be chosen with use_key_set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from . import client as _client   # looked up at call time, so tests can patch client.load_env
from .client import GEMINI_BASE_URL, OpenAICompatClient, client_for


@dataclass(frozen=True)
class Provider:
    label: str
    key_var: str
    base_var: str
    default_base: str | None = None     # None: the base URL variable must be set
    model_var: str | None = None
    default_model: str | None = None    # None: keep the config's model names
    reasoning_effort: str | None = None
    default_key: str | None = None      # for servers that ignore the key, such as Ollama
    old_key_var: str | None = None      # an earlier name for the key, still read so old .env files work


PROVIDERS = {
    "gemini": Provider("Gemini", "GEMINI_API_KEY", "GEMINI_BASE_URL", GEMINI_BASE_URL,
                       reasoning_effort="minimal", old_key_var="LLM_API_KEY"),
    # gpt-oss accepts low / medium / high, not Gemini's "minimal"
    "groq": Provider("Groq", "GROQ_API_KEY", "GROQ_BASE_URL", "https://api.groq.com/openai/v1",
                     "GROQ_MODEL", "openai/gpt-oss-20b", reasoning_effort="low"),
    # llama3 has no reasoning setting, so none is sent
    "ollama": Provider("Ollama", "OLLAMA_API_KEY", "OLLAMA_BASE_URL", "http://localhost:11434/v1",
                       "OLLAMA_MODEL", "llama3", default_key="ollama"),
}


def _get(name: str) -> Provider:
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}")
    return PROVIDERS[name]


def _endpoint(p: Provider) -> tuple[str | None, str | None]:
    _client.load_env()
    key = os.environ.get(p.key_var) or (os.environ.get(p.old_key_var) if p.old_key_var else None)
    return os.environ.get(p.base_var) or p.default_base, key or p.default_key


def use_key_set(name: str, providers: list[str]) -> list[str]:
    """Point each provider at its key from set `name` (GROQ_API_KEY_<name>, GEMINI_API_KEY_<name>), so
    several runs can share the work across accounts. A provider with no key in the set keeps its
    usual key. Returns the variables that were switched."""
    _client.load_env()
    switched = []
    for p in {_get(n) for n in providers}:
        key = os.environ.get(f"{p.key_var}_{name}")
        if key:
            os.environ[p.key_var] = key
            switched.append(p.key_var)
    return sorted(switched)


def provider_status(name: str) -> tuple[bool, str]:
    """(ready, message): is this provider's endpoint and key configured? Never raises."""
    try:
        p = _get(name)
        base, key = _endpoint(p)
    except Exception as e:
        return False, f"{name}: {e}"
    if base and key:
        return True, f"{p.label} endpoint and key found."
    missing = [v for v, ok in ((p.base_var, base), (p.key_var, key)) if not ok]
    return False, (f"No {p.label} API key found. Set {' and '.join(missing)} in the environment or in "
                   "the project's .env file, or choose another mode.")


def provider_client(name: str) -> OpenAICompatClient:
    p = _get(name)
    base, key = _endpoint(p)
    if not base or not key:
        raise RuntimeError(provider_status(name)[1])
    return client_for(base, key)


def provider_model(name: str) -> str | None:
    """The model every role uses with this provider, or None to keep the config's names."""
    p = _get(name)
    _client.load_env()
    return (os.environ.get(p.model_var) if p.model_var else None) or p.default_model


def apply_provider(config, name: str):
    """`config` with every model role set to the provider's model (unchanged for gemini),
    so the saved result records which model actually ran."""
    model = provider_model(name)
    if model is None:
        return config
    models = replace(config.models, interviewer=model, diagnoser=model, topic_extractor=model, grader=model)
    return config.with_(models=models)
