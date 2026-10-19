import numpy as np
import pytest

from part1.bank import load_bank
from part1.index import QuestionIndex, find_near_duplicates

from conftest import fake_embed


@pytest.fixture
def index(bank):
    return QuestionIndex.build(bank, embed_fn=fake_embed)


def ids(results):
    return [q["id"] for q, _ in results]


def test_build_has_one_row_per_question(bank, index):
    assert index.embeddings.shape[0] == len(bank.questions) == 90
    assert index.ids == sorted(index.ids)


def test_save_and_load_roundtrip(bank, index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = QuestionIndex.load(tmp_path / "idx", bank=bank, embed_fn=fake_embed)
    assert loaded.ids == index.ids
    assert np.allclose(loaded.embeddings, index.embeddings)
    assert loaded.bank_hash == bank.content_hash()
    assert not loaded.is_stale(bank)


def test_stale_index_is_detected(bank, index, tmp_path, recwarn):
    index.save(tmp_path / "idx")
    bank.get_question("PY-L1-001")["question"] = "A completely different question."
    assert index.is_stale(bank)
    QuestionIndex.load(tmp_path / "idx", bank=bank, embed_fn=fake_embed)
    assert any("re-run" in str(w.message) for w in recwarn)


def test_status_change_does_not_make_index_stale(bank, index):
    bank.get_question("PY-L1-001")["status"] = "checked"
    assert not index.is_stale(bank)


def test_load_fails_clearly_when_questions_are_gone(bank, index, tmp_path):
    index.save(tmp_path / "idx")
    bank.questions = [q for q in bank.questions if q["id"] != "PY-L1-001"]
    bank._index()
    with pytest.raises(ValueError, match="stale"):
        QuestionIndex.load(tmp_path / "idx", bank=bank, embed_fn=fake_embed)


def test_filter_by_skill_and_levels(index):
    results = index.search(skill="sql", levels=[2, 3], k=100)
    assert len(results) == 20
    assert all(q["skill"] == "sql" and q["level"] in (2, 3) for q, _ in results)


def test_exclude_ids(index):
    first = ids(index.search(skill="py", levels=[1], k=3))
    again = ids(index.search(skill="py", levels=[1], exclude_ids=set(first), k=3))
    assert not set(first) & set(again)
    assert len(again) == 3


def test_k_limits_and_empty_result(index):
    assert len(index.search(k=2)) == 2
    assert index.search(skill="py", levels=[1], exclude_ids={f"PY-L1-{i:03d}" for i in range(1, 11)}) == []


def test_prefer_topics_rank_first(index):
    results = index.search(skill="py", prefer_topics=["py-alpha", "py-beta"], k=100)
    overlaps = [len({"py-alpha", "py-beta"} & set(q["topics"])) for q, _ in results]
    assert overlaps == sorted(overlaps, reverse=True)
    assert overlaps[0] == 2
    assert results[0][1] == 2.0   # without a query the score is the overlap count


def test_filters_apply_before_ranking(index):
    # best topic matches are at other levels; filtering must still restrict to level 1
    results = index.search(skill="py", levels=[1], prefer_topics=["py-alpha"], k=100)
    assert {q["level"] for q, _ in results} == {1}


def test_query_ranks_by_similarity(bank, index):
    target = bank.get_question("DS-L2-003")
    results = index.search(query=target["question"], skill="ds", k=3)
    assert results[0][0]["id"] == "DS-L2-003"
    assert results[0][1] == pytest.approx(1.0, abs=0.5)  # a long text, so not exactly 1
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_topics_take_priority_over_similarity(bank, index):
    target = bank.get_question("PY-L1-001")
    results = index.search(query=target["question"], skill="py",
                           prefer_topics=["py-epsilon"], k=100)
    overlaps = [("py-epsilon" in q["topics"]) for q, _ in results]
    assert overlaps == sorted(overlaps, reverse=True)


def test_ties_are_deterministic(index):
    assert ids(index.search(skill="py", k=10)) == ids(index.search(skill="py", k=10))
    assert ids(index.search(skill="py", k=3)) == ["PY-L1-001", "PY-L1-002", "PY-L1-003"]


def test_near_duplicates(bank):
    a = bank.get_question("PY-L1-001")
    b = bank.get_question("PY-L3-004")
    for key in ("question", "topics", "key_points"):
        b[key] = a[key]
    index = QuestionIndex.build(bank, embed_fn=fake_embed)
    pairs = index.near_duplicates(threshold=0.999)
    assert [(x, y) for x, y, _ in pairs] == [("PY-L1-001", "PY-L3-004")]
    assert pairs[0][2] > 0.999


def test_find_near_duplicates_threshold():
    emb = np.array([[1, 0], [1, 0.1], [0, 1]], dtype=np.float32)
    assert [(a, b) for a, b, _ in find_near_duplicates(["a", "b", "c"], emb, 0.9)] == [("a", "b")]
    assert find_near_duplicates(["a", "b", "c"], emb, 0.999) == []
    assert find_near_duplicates(["a"], emb[:1]) == []


def test_empty_bank_builds_empty_index(tmp_path):
    empty = load_bank(tmp_path / "none")
    index = QuestionIndex.build(empty, embed_fn=fake_embed)
    assert index.search(query="anything", k=3) == []
