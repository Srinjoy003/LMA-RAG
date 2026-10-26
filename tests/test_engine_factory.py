"""The component factory: stubs offline, prompted components live (fakes, no network)."""

import sys

import pytest

from engine import factory
from engine.factory import FactoryError, build_components, grader_model, live_status
from shared.interfaces import SYSTEMS, ModelSettings
from shared.stubs import TagDiagnoser, TagGrader

from engine_support import STRONG, assert_valid, job_for, run


def test_offline_components_are_the_stubs():
    config = SYSTEMS["A2"]
    components = build_components(config, live=False)
    assert isinstance(components.grader, TagGrader) and components.grader.votes == config.grader_votes
    assert isinstance(components.diagnoser, TagDiagnoser)
    assert components.llm is None and components.topic_extractor is None
    assert "sql.joins.left-join-drops-unmatched" in components.misconceptions
    found = components.question_source.search(skill="py", levels=[2], k=3)
    assert len(found) == 3 and all(q["skill"] == "py" and q["level"] == 2 for q, _ in found)


def test_offline_components_run_an_interview():
    components = build_components(SYSTEMS["A2"], live=False)
    result, _ = run(STRONG, "A2", components=components)
    assert result["errors"] == [] and result["report"]["decision"] == "fit"
    assert_valid(result)


def test_offline_without_a_saved_index_still_works(tmp_path):
    import shutil
    shutil.copytree(factory.ROOT / "data", tmp_path / "data")
    components = build_components(SYSTEMS["A2"], live=False, root=tmp_path)
    assert components.question_source.search(skill="sql", levels=[1], k=2)


def test_grader_model_uses_a_real_name_else_the_fallback():
    assert grader_model(SYSTEMS["A2"]) == "gemini-3.5-flash-lite"            # "finetuned-grader" is a placeholder
    assert grader_model(SYSTEMS["A3"]) == "gemini-3.5-flash-lite"
    custom = SYSTEMS["A2"].with_(models=ModelSettings(grader="my-model"))
    assert grader_model(custom) == "my-model"


# ---- live path with fakes ---------------------------------------------------------------------

class FakeGrader:
    def __init__(self, llm, model, votes=3, **kw):
        self.llm, self.model, self.votes = llm, model, votes


class FakeDiagnoser:
    def __init__(self, llm, model, misconceptions=None, votes=3, **kw):
        self.llm, self.model, self.misconceptions, self.votes = llm, model, misconceptions, votes


class FakeExtractor:
    def __init__(self, llm, model):
        self.llm, self.model = llm, model

    def __call__(self, description):
        return ["py-pandas"]


@pytest.fixture
def fake_live(monkeypatch):
    llm = object()
    monkeypatch.setattr("llm.client.get_client", lambda role=None: llm)
    monkeypatch.setattr("part3.grader.PromptedGrader", FakeGrader)
    monkeypatch.setattr("part3.diagnoser.StructuredDiagnoser", FakeDiagnoser)
    monkeypatch.setattr("part1.extract_topics.LLMTopicExtractor", FakeExtractor, raising=False)
    return llm


def test_live_components_are_built_from_the_config(fake_live):
    config = SYSTEMS["A4"].with_(models=ModelSettings(diagnoser="diag-model", topic_extractor="topic-model"),
                                 grader_votes=5)
    components = build_components(config, live=True)
    assert components.llm.inner is fake_live      # wrapped in MeteredLLM
    assert isinstance(components.grader, FakeGrader)
    assert (components.grader.model, components.grader.votes) == ("gemini-3.5-flash-lite", 5)
    assert components.grader.llm is components.llm
    assert isinstance(components.diagnoser, FakeDiagnoser)
    assert (components.diagnoser.model, components.diagnoser.votes) == ("diag-model", 1)       # A4 uses one diagnosis vote
    assert components.diagnoser.misconceptions is components.misconceptions
    assert "sql.joins.left-join-drops-unmatched" in components.misconceptions
    assert isinstance(components.topic_extractor, FakeExtractor)
    assert components.topic_extractor.model == "topic-model"


def test_live_uses_a_real_grader_model_name_as_given(fake_live):
    config = SYSTEMS["A2"].with_(models=ModelSettings(grader="tuned-v1"))
    assert build_components(config, live=True).grader.model == "tuned-v1"


def test_live_without_topic_extraction_has_no_extractor(fake_live):
    components = build_components(SYSTEMS["A2"].with_(use_topic_extraction=False), live=True)
    assert components.topic_extractor is None


def test_live_without_an_extractor_class_falls_back_to_jobs_json(fake_live, monkeypatch):
    monkeypatch.delattr("part1.extract_topics.LLMTopicExtractor", raising=False)
    with pytest.warns(UserWarning, match="LLMTopicExtractor"):
        components = build_components(SYSTEMS["A2"], live=True)
    assert components.topic_extractor is None and isinstance(components.grader, FakeGrader)


@pytest.mark.parametrize("module", ["llm.client", "part3.grader", "part3.diagnoser", "part1.extract_topics"])
def test_a_missing_live_module_gives_a_clear_error(fake_live, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, None)        # makes the import fail
    with pytest.raises(FactoryError, match=module.replace(".", r"\.")):
        build_components(SYSTEMS["A2"], live=True)


def test_a_module_that_fails_while_importing_gives_a_clear_error(fake_live, monkeypatch):
    def boom(name, *a, **k):
        raise SyntaxError("broken module")

    monkeypatch.setattr(factory.importlib, "import_module", boom)
    with pytest.raises(FactoryError, match="SyntaxError"):
        build_components(SYSTEMS["A2"], live=True)


def test_live_without_a_key_gives_a_clear_error(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("llm.client.load_env", lambda path=None: {})
    with pytest.raises(FactoryError, match="LLM_API_KEY"):
        build_components(SYSTEMS["A2"], live=True)


def test_live_status_reports_a_missing_key(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("llm.client.load_env", lambda path=None: {})
    ok, message = live_status()
    assert ok is False and "API key" in message
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    assert live_status()[0] is True


def test_metered_llm_counts_calls_and_cache_hits():
    from engine.factory import MeteredLLM, usage_delta

    class Inner:
        def __init__(self):
            self.n = 0

        def complete(self, messages, **kwargs):
            self.n += 1
            return {"text": "x", "model": kwargs["model"], "input_tokens": 10, "output_tokens": 2,
                    "cached": self.n == 2}

    llm = MeteredLLM(Inner())
    before = llm.usage_totals()
    llm.complete([], model="m1")
    llm.complete([], model="m1")
    llm.complete([], model="m2")
    delta = usage_delta(before, llm.usage_totals())
    assert delta["llm_calls"] == 3 and delta["api_calls"] == 2 and delta["input_tokens"] == 30
    assert delta["by_model"]["m1"]["llm_calls"] == 2 and delta["by_model"]["m2"]["api_calls"] == 1


def test_save_result_keeps_agent_and_seeded_runs_apart(tmp_path):
    from engine.result import save_result
    base = {"system": "A2", "job_id": "j", "candidate_id": "CAND-001", "started_at": "2026-10-19T00:00:00Z"}
    a = save_result({**base, "config": {"driver": "code", "seed": 0}}, tmp_path)
    b = save_result({**base, "config": {"driver": "agent", "seed": 0}}, tmp_path)
    c = save_result({**base, "config": {"driver": "code", "seed": 3}}, tmp_path)
    assert [p.name for p in (a, b, c)] == ["CAND-001.json", "CAND-001-agent.json", "CAND-001-seed3.json"]
