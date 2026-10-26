import pytest

from fake_llm import FakeLLM
from part1.extract_topics import (DEFAULT_MODEL, LLMTopicExtractor, extract_topics, get_model,
                                  score_topics)
from shared.interfaces import ModelSettings, TopicExtractor

TOPICS = {
    "py": [{"id": "py-collections", "name": "Lists and dicts"}, {"id": "py-oop", "name": "Classes"}],
    "sql": [{"id": "sql-joins", "name": "Joins"}],
}


def answer(*topics):
    return {"topics": list(topics)}


def test_returns_topics_from_model_answer():
    llm = FakeLLM([answer("sql-joins", "py-collections")])
    assert extract_topics("Join tables.", llm=llm, topics=TOPICS) == ["sql-joins", "py-collections"]


def test_duplicates_are_removed():
    llm = FakeLLM([answer("py-oop", "sql-joins", "py-oop")])
    assert extract_topics("job", llm=llm, topics=TOPICS) == ["py-oop", "sql-joins"]


def test_unknown_topics_are_rejected_by_the_schema_or_dropped():
    # The schema enum makes the fake (like the real client) reject an unknown id...
    with pytest.raises(Exception, match="schema"):
        extract_topics("job", llm=FakeLLM([answer("py-oop", "made-up-topic")]), topics=TOPICS)

    # ...and if a client ever lets one through, it is dropped anyway.
    class Lenient:
        def complete(self, messages, **kw):
            return {"parsed": answer("py-oop", "made-up-topic", "sql-joins"), "text": ""}

    assert extract_topics("job", llm=Lenient(), topics=TOPICS) == ["py-oop", "sql-joins"]


def test_request_constrains_output_to_topic_ids():
    llm = FakeLLM([answer()])
    extract_topics("Some job description.", llm=llm, topics=TOPICS)
    call = llm.calls[0]
    schema = call["json_schema"]
    assert schema["properties"]["topics"]["items"]["enum"] == ["py-collections", "py-oop", "sql-joins"]
    system, user = (m["content"] for m in call["messages"])
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "Some job description." in user and "py-collections" in system
    assert call["temperature"] == 0.0 and call["tools"] is None


def test_default_model_and_env_override(monkeypatch):
    monkeypatch.delenv("TOPIC_MODEL", raising=False)
    llm = FakeLLM([answer()])
    extract_topics("job", llm=llm, topics=TOPICS)
    assert llm.calls[0]["model"] == DEFAULT_MODEL == ModelSettings().topic_extractor

    monkeypatch.setenv("TOPIC_MODEL", "some-other-model")
    assert get_model() == "some-other-model"
    llm = FakeLLM([answer()])
    extract_topics("job", llm=llm, topics=TOPICS)
    assert llm.calls[0]["model"] == "some-other-model"

    llm = FakeLLM([answer()])
    extract_topics("job", llm=llm, model="explicit", topics=TOPICS)
    assert llm.calls[0]["model"] == "explicit"


def test_empty_description_makes_no_call():
    llm = FakeLLM([answer("py-oop")])
    assert extract_topics("   ", llm=llm, topics=TOPICS) == []
    assert llm.calls == []


def test_unparsed_answer_raises_clear_error():
    class NoJson:
        def complete(self, messages, **kw):
            return {"text": "not json"}

    with pytest.raises(RuntimeError, match="could not parse"):
        extract_topics("job", llm=NoJson(), topics=TOPICS)


def test_missing_credentials_raise(monkeypatch):
    from llm import client
    monkeypatch.setattr(client, "load_env", lambda *a, **k: {})
    monkeypatch.setattr(client, "_clients", {})
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY_TOPIC_EXTRACTOR", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        extract_topics("job", topics=TOPICS)


def test_uses_real_topics_file_by_default():
    llm = FakeLLM([answer("py-pandas", "sql-window-functions")])
    assert extract_topics("job", llm=llm) == ["py-pandas", "sql-window-functions"]
    assert len(llm.calls[0]["json_schema"]["properties"]["topics"]["items"]["enum"]) == 27


def test_llm_topic_extractor_matches_the_protocol():
    llm = FakeLLM([answer("sql-joins")])
    extractor = LLMTopicExtractor(llm, "my-model", topics=TOPICS)
    assert isinstance(extractor, TopicExtractor)
    assert extractor("Join some tables") == ["sql-joins"]
    assert llm.calls[0]["model"] == "my-model"


def test_score_topics():
    s = score_topics(["a", "b", "c"], ["b", "c", "d", "e"])
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert s["missed"] == ["d", "e"] and s["extra"] == ["a"]
    assert score_topics([], ["a"])["f1"] == 0.0
