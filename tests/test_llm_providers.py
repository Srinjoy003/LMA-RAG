"""Named providers (llm.providers) and the factory's provider path. No network."""

import pytest

from engine.factory import FactoryError, build_components, live_status
from llm import providers
from shared.interfaces import SYSTEMS

ENV = ("LLM_API_KEY", "LLM_BASE_URL", "GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL",
       "OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("llm.client.load_env", lambda path=None: {})
    return monkeypatch


def test_status_names_the_missing_variable(clean_env):
    ok, message = providers.provider_status("groq")
    assert ok is False and "GROQ_API_KEY" in message and "GROQ_BASE_URL" not in message   # base has a default
    ok, message = providers.provider_status("gemini")
    assert ok is False and "LLM_API_KEY" in message and "LLM_BASE_URL" in message
    clean_env.setenv("GROQ_API_KEY", "test-key")
    assert providers.provider_status("groq")[0] is True
    assert live_status("groq")[0] is True and live_status("nope")[0] is False


def test_client_uses_the_provider_endpoint(clean_env):
    clean_env.setenv("GROQ_API_KEY", "test-key")
    client = providers.provider_client("groq")
    assert client.base_url.startswith("https://api.groq.com/openai/v1")
    assert providers.provider_client("groq") is client          # shared per endpoint and key
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        providers.provider_client("gemini")


def test_a_key_set_switches_only_the_providers_it_has_a_key_for(clean_env):
    clean_env.setenv("GROQ_API_KEY", "mine")
    clean_env.setenv("LLM_API_KEY", "mine-too")
    clean_env.setenv("GROQ_API_KEY_2", "friends")
    assert providers.use_key_set("2", ["gemini", "groq"]) == ["GROQ_API_KEY"]
    assert providers.provider_client("groq")._api_key == "friends"
    assert providers.use_key_set("3", ["groq"]) == []


def test_ollama_needs_no_key_and_sends_no_reasoning_effort(clean_env):
    assert providers.provider_status("ollama")[0] is True
    assert providers.provider_client("ollama").base_url == "http://localhost:11434/v1"
    assert providers.PROVIDERS["ollama"].reasoning_effort is None
    assert providers.apply_provider(SYSTEMS["A2"], "ollama").models.grader == "llama3"


def test_apply_provider_sets_every_role_except_for_gemini(clean_env):
    config = SYSTEMS["A2"]
    assert providers.apply_provider(config, "gemini") == config
    groq = providers.apply_provider(config, "groq").models
    assert {groq.interviewer, groq.diagnoser, groq.topic_extractor, groq.grader} == {"openai/gpt-oss-20b"}
    assert groq.simulator == config.models.simulator
    clean_env.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    assert providers.apply_provider(config, "groq").models.grader == "openai/gpt-oss-120b"


class Fake:
    def __init__(self, llm, model, *args, **kw):
        self.llm, self.model, self.kw = llm, model, kw

    def __call__(self, description):
        return []


def test_factory_builds_with_the_provider_client_and_effort(clean_env):
    llm = object()
    clean_env.setattr("llm.providers.provider_client", lambda name: llm)
    for target in ("part3.grader.PromptedGrader", "part3.diagnoser.StructuredDiagnoser",
                   "part1.extract_topics.LLMTopicExtractor"):
        clean_env.setattr(target, Fake)
    config = providers.apply_provider(SYSTEMS["A2"], "groq")
    components = build_components(config, live=True, provider="groq")
    assert components.llm.inner is llm
    for part in (components.grader, components.diagnoser, components.topic_extractor):
        assert part.model == "openai/gpt-oss-20b" and part.kw["reasoning_effort"] == "low"


def test_factory_reports_a_missing_provider_key(clean_env):
    with pytest.raises(FactoryError, match="GROQ_API_KEY"):
        build_components(SYSTEMS["A2"], live=True, provider="groq")
