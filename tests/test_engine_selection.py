import random
from collections import Counter

import pytest

from engine.selection import (InterviewState, check_stop, choose_next, has_next, next_target,
                              record_asked)
from shared.interfaces import SYSTEMS, Components
from shared.stubs import TagGrader

from engine_support import FakeGenerator, make_components


def q(qid, skill, level, topics, misconceptions=()):
    return {"id": qid, "skill": skill, "level": level, "topics": list(topics),
            "misconceptions": list(misconceptions), "question": f"Question {qid}", "source": "generated"}


class ListSource:
    """A hand-made bank. It filters like the real index but returns questions in reverse
    id order, so the engine's own ranking has to do the work."""

    def __init__(self, questions):
        self.questions = questions

    def search(self, query=None, skill=None, levels=None, prefer_topics=None,
               exclude_ids=frozenset(), k=5):
        found = [(x, 0.0) for x in sorted(self.questions, key=lambda x: x["id"], reverse=True)
                 if (skill is None or x["skill"] == skill)
                 and (levels is None or x["level"] in levels) and x["id"] not in exclude_ids]
        return found[:k]


def grid_bank(per_cell=3):
    """per_cell questions for every skill and level, all on one neutral topic per skill."""
    return [q(f"{s.upper()}-L{lv}-{i}", s, lv, [f"{s}-other"])
            for s in ("py", "sql", "ds") for lv in (1, 2, 3) for i in range(per_cell)]


def make_state(questions, system="A2", required=(), seed=0, generator=None, **changes):
    config = SYSTEMS[system].with_(seed=seed, **changes)
    components = Components(question_source=ListSource(questions), grader=TagGrader(),
                            question_generator=generator)
    job = {"id": "job", "min_levels": {"py": 2, "sql": 2, "ds": 2}}
    return InterviewState(job, config, components, list(required), random.Random(seed))


def ask(state):
    """Choose the next question and mark it as asked and answered."""
    choice = choose_next(state)
    record_asked(state, choice)
    state.pending = None
    return choice


# ---- skill and level -------------------------------------------------------

def test_first_skill_on_a_tie_follows_py_sql_ds():
    state = make_state(grid_bank())
    assert choose_next(state)["skill"] == "py"
    state.exhausted.add("py")
    assert choose_next(state)["skill"] == "sql"
    state.exhausted.add("sql")
    assert choose_next(state)["skill"] == "ds"


def test_highest_se_skill_goes_first():
    state = make_state(grid_bank())
    state.scores["py"].update(2, 1)       # py becomes more certain than the others
    assert choose_next(state)["skill"] == "sql"
    state.scores["sql"].update(2, 1)
    state.scores["sql"].update(2, 1)      # now sql is the most certain; ds is the least
    assert choose_next(state)["skill"] == "ds"


def test_exhausted_skills_are_skipped_even_if_most_uncertain():
    state = make_state(grid_bank())
    state.exhausted.add("py")
    assert choose_next(state)["skill"] == "sql"


@pytest.mark.parametrize("score,level", [(None, 2), (2, 3), (0, 1)])
def test_level_nearest_the_estimate(score, level):
    state = make_state(grid_bank())
    for skill in ("sql", "ds"):
        for _ in range(8):
            state.scores[skill].update(2, 1)     # certain, so py has the highest SE
    if score is not None:
        state.scores["py"].update(2, score)
    assert next_target(state)[:2] == ("py", level)


def test_first_question_is_level_2():
    assert next_target(make_state(grid_bank()))[:2] == ("py", 2)


# ---- question ranking --------------------------------------------------------

def test_job_aware_prefers_questions_on_untested_required_topics():
    bank = [q("PY-L2-a", "py", 2, ["py-other"]), q("PY-L2-b", "py", 2, ["py-pandas"]),
            q("PY-L2-c", "py", 2, ["py-other"])]
    for seed in range(5):
        state = make_state(bank, required=["py-pandas"], seed=seed)
        choice = choose_next(state)
        assert choice["question"]["id"] == "PY-L2-b"
        assert "py-pandas" in choice["reason"]


def test_job_aware_forgets_a_topic_once_it_is_tested():
    bank = [q("PY-L2-a", "py", 2, ["py-pandas"]), q("PY-L2-b", "py", 2, ["py-collections"]),
            q("PY-L2-c", "py", 2, ["py-pandas"])]
    state = make_state(bank, required=["py-pandas", "py-collections"])
    record_asked(state, {"question": bank[0]})        # a pandas question was asked
    state.pending = None
    state.scores["py"].update(2, 1)
    state.scores["py"].update(2, 1)
    state.exhausted.update({"sql", "ds"})
    assert choose_next(state)["question"]["id"] == "PY-L2-b"   # the other required topic, not more pandas


def test_job_blind_ignores_required_topics():
    bank = [q("PY-L2-a", "py", 2, ["py-other"]), q("PY-L2-b", "py", 2, ["py-pandas"]),
            q("PY-L2-c", "py", 2, ["py-other"])]
    picked = Counter(choose_next(make_state(bank, "A5", required=["py-pandas"], seed=s))["question"]["id"]
                     for s in range(40))
    assert set(picked) == {"PY-L2-a", "PY-L2-b", "PY-L2-c"}


def test_suspected_misconception_beats_an_ordinary_question():
    bank = [q("PY-L2-a", "py", 2, ["py-other"]), q("PY-L2-b", "py", 2, ["py-other"], ["py.collections.tuple-mutable"]),
            q("PY-L2-c", "py", 2, ["py-other"])]
    for seed in range(5):
        state = make_state(bank, seed=seed)
        state.tracker.record("py.collections.tuple-mutable", 1)    # seen once: suspected
        assert choose_next(state)["question"]["id"] == "PY-L2-b"
    state = make_state(bank)
    assert state.tracker.suspected() == set()


def test_required_topic_beats_suspected_misconception_without_probing():
    """The original ranking (B1, or probe_missed_topics off): job topics first."""
    bank = [q("PY-L2-a", "py", 2, ["py-pandas"]), q("PY-L2-b", "py", 2, ["py-other"], ["py.collections.tuple-mutable"])]
    state = make_state(bank, required=["py-pandas"], probe_missed_topics=False)
    state.tracker.record("py.collections.tuple-mutable", 1)
    assert choose_next(state)["question"]["id"] == "PY-L2-a"


def test_suspected_misconception_beats_required_topic_with_probing():
    bank = [q("PY-L2-a", "py", 2, ["py-pandas"]), q("PY-L2-b", "py", 2, ["py-other"], ["py.collections.tuple-mutable"])]
    state = make_state(bank, required=["py-pandas"])
    state.tracker.record("py.collections.tuple-mutable", 1)
    assert choose_next(state)["question"]["id"] == "PY-L2-b"


# ---- probing missed topics (spec section 4) -----------------------------------------------

M1, M2, M3 = "py.collections.tuple-mutable", "py.collections.dict-order", "py.collections.set-index"


def graded(state, qid, score, confirmed=None):
    """Ask question `qid` of the state's bank, give it a final grade, optionally with a
    misconception confirmed on it."""
    question = next(x for x in state.source.questions if x["id"] == qid)
    record_asked(state, {"question": question})
    state.pending = None
    if confirmed:
        state.tracker.record(confirmed, len(state.asked))
        state.tracker.record(confirmed, len(state.asked))
    state.graded.append({"turn": len(state.asked), "skill": question["skill"],
                         "topics": question["topics"], "score": score})
    state.scores[question["skill"]].update(question["level"], score)


def probe_bank():
    return [q("PY-L1-w", "py", 1, ["py-collections"], [M1]),        # asked first, wrong
            q("PY-L2-a", "py", 2, ["py-collections"], [M2]),        # same topic, new misconception
            q("PY-L2-b", "py", 2, ["py-other"], [M3]),              # other topic, new misconception
            q("PY-L2-c", "py", 2, ["py-collections"], [M1]),        # same topic, already probed
            q("PY-L2-d", "py", 2, ["py-collections"], [])]          # same topic, lists nothing


def test_probe_picks_the_weak_topic_with_an_unprobed_misconception():
    for seed in range(6):
        state = make_state(probe_bank(), seed=seed)
        graded(state, "PY-L1-w", 0)
        choice = choose_next(state)
        assert choice["question"]["id"] == "PY-L2-a"
        assert "probes a topic" in choice["reason"] and M2 in choice["reason"]


def test_no_probe_after_a_correct_answer():
    picked = set()
    for seed in range(12):
        state = make_state(probe_bank(), seed=seed)
        graded(state, "PY-L1-w", 2)
        picked.add(choose_next(state)["question"]["id"])
    assert picked == {"PY-L2-a", "PY-L2-b"}      # tier (d): both list one unprobed misconception
    state = make_state(probe_bank())
    graded(state, "PY-L1-w", 2)
    assert "probes a topic" not in choose_next(state)["reason"]


def test_no_probe_when_the_misconception_on_that_question_was_confirmed():
    for seed in range(6):
        state = make_state(probe_bank(), seed=seed)
        graded(state, "PY-L1-w", 0, confirmed=M1)
        assert "probes a topic" not in choose_next(state)["reason"]
    # one reply only = suspected, not confirmed: the topic is still probed (and (a) wins if it is listed)
    state = make_state(probe_bank())
    graded(state, "PY-L1-w", 0)
    state.tracker.record(M1, 1)
    assert choose_next(state)["question"]["id"] == "PY-L2-c"      # (a) lists the suspected M1


def test_suspected_beats_probe_and_probe_beats_job_topic():
    bank = probe_bank() + [q("PY-L2-e", "py", 2, ["py-pandas"], [])]
    state = make_state(bank, required=["py-pandas"])
    graded(state, "PY-L1-w", 0)
    assert choose_next(state)["question"]["id"] == "PY-L2-a"      # probe beats the required topic
    state.tracker.record(M3, 1)                                   # M3 suspected
    assert choose_next(state)["question"]["id"] == "PY-L2-b"      # suspected beats probe


def test_job_topic_beats_more_unprobed_misconceptions():
    bank = [q("PY-L2-a", "py", 2, ["py-pandas"], []), q("PY-L2-b", "py", 2, ["py-other"], [M1, M2])]
    for seed in range(5):
        state = make_state(bank, required=["py-pandas"], seed=seed)
        assert choose_next(state)["question"]["id"] == "PY-L2-a"
    state = make_state(bank, required=["py-pandas"], retrieval="job-blind")
    choice = choose_next(state)
    assert choice["question"]["id"] == "PY-L2-b" and "not yet probed" in choice["reason"]


def test_more_unprobed_misconceptions_win_the_last_tier():
    bank = [q("PY-L2-a", "py", 2, ["py-other"], [M1]), q("PY-L2-b", "py", 2, ["py-other"], [M2, M3]),
            q("PY-L2-c", "py", 2, ["py-other"], [])]
    for seed in range(5):
        assert choose_next(make_state(bank, seed=seed))["question"]["id"] == "PY-L2-b"


def test_probed_misconceptions_are_those_listed_on_asked_questions():
    state = make_state(probe_bank())
    assert state.probed_misconceptions() == set()
    graded(state, "PY-L1-w", 2)
    assert state.probed_misconceptions() == {M1}


def test_b1_ranking_ignores_probing():
    """B1 keeps the original three-tier ranking, so the same seed gives the same questions
    whether or not misconceptions were missed."""
    bank = probe_bank() + [q("PY-L2-e", "py", 2, ["py-pandas"], [])]
    for seed in range(5):
        state = make_state(bank, "B1", required=["py-pandas"], seed=seed)
        assert state.config.probe_missed_topics is False
        graded(state, "PY-L1-w", 0)
        state.exhausted.update({"sql", "ds"})
        assert choose_next(state)["question"]["id"] == "PY-L2-e"      # job topic, no probe


def test_a_confirmed_misconception_is_not_chased_again():
    bank = [q("PY-L2-a", "py", 2, ["py-other"]), q("PY-L2-b", "py", 2, ["py-other"], ["py.collections.tuple-mutable"])]
    picked = Counter()
    for seed in range(30):
        state = make_state(bank, seed=seed, probe_missed_topics=False)
        for turn in (1, 2):
            state.tracker.record("py.collections.tuple-mutable", turn)
        assert state.tracker.suspected() == set()
        picked[choose_next(state)["question"]["id"]] += 1
    assert picked["PY-L2-a"] > 0 and picked["PY-L2-b"] > 0     # neither is preferred


def test_ties_are_broken_by_seed_not_by_the_sources_order():
    bank = [q(f"PY-L2-{i}", "py", 2, ["py-other"]) for i in range(8)]
    same = [choose_next(make_state(bank, seed=3))["question"]["id"] for _ in range(3)]
    assert len(set(same)) == 1
    assert len({choose_next(make_state(bank, seed=s))["question"]["id"] for s in range(30)}) > 3


def test_asked_questions_are_not_asked_again():
    state = make_state(grid_bank(per_cell=2))
    ids = []
    for _ in range(6):
        ids.append(ask(state)["question"]["id"])
    assert len(set(ids)) == 6


# ---- fallbacks ------------------------------------------------------------------

@pytest.mark.parametrize("score,nearer", [(2, 3), (0, 1)])
def test_fallback_goes_to_the_adjacent_level_nearer_the_estimate(score, nearer):
    bank = [q("PY-L1-a", "py", 1, ["py-other"]), q("PY-L3-a", "py", 3, ["py-other"])]
    state = make_state(bank)
    state.exhausted.update({"sql", "ds"})
    state.scores["py"].update(2, score, 0.4)     # a small move: still level 2, but leaning one way
    assert next_target(state)[1] == 2
    choice = choose_next(state)
    assert choice["target_level"] == 2 and choice["level"] == nearer


def test_fallback_uses_the_other_level_when_the_nearer_one_is_empty():
    bank = [q("PY-L1-a", "py", 1, ["py-other"])]
    state = make_state(bank)
    state.exhausted.update({"sql", "ds"})
    choice = choose_next(state)
    assert choice["target_level"] == 2 and choice["level"] == 1
    assert "no unasked level-2 questions left" in choice["reason"]


def test_generated_question_when_the_bank_is_empty():
    generator = FakeGenerator()
    state = make_state([], required=["py-pandas", "sql-joins"], generator=generator)
    choice = choose_next(state)
    assert choice["source"] == "generated" and choice["skill"] == "py"
    assert choice["question"]["topics"] == ["py-pandas"]
    assert generator.calls == [("py", 2, "py-pandas")]


def test_generated_question_needs_an_untested_required_topic_of_that_skill():
    generator = FakeGenerator()
    state = make_state([], required=["sql-joins"], generator=generator)
    choice = choose_next(state)
    assert choice["skill"] == "sql"            # py has nothing left and no required topic
    assert "py" in state.exhausted


def test_no_generated_question_when_not_allowed_or_no_generator():
    for state in (make_state([], required=["py-pandas"], generator=FakeGenerator(), allow_generated_questions=False),
                  make_state([], required=["py-pandas"], generator=None)):
        assert choose_next(state) is None
        assert state.exhausted == {"py", "sql", "ds"}
        assert check_stop(state) == "bank_exhausted"


def test_a_failing_generator_marks_the_skill_exhausted():
    class Broken:
        def generate(self, skill, level, topic):
            raise RuntimeError("no key")

    state = make_state([], required=["py-pandas"], generator=Broken())
    assert choose_next(state) is None
    assert any("generator failed" in e for e in state.errors)


# ---- fixed mode --------------------------------------------------------------------

def test_fixed_mode_rotates_skills_with_levels_1_2_2_3():
    state = make_state(grid_bank(per_cell=4), "B1")
    asked = [(c["skill"], c["level"]) for c in (ask(state) for _ in range(12))]
    assert [s for s, _ in asked] == ["py", "sql", "ds"] * 4
    for skill in ("py", "sql", "ds"):
        assert [lv for s, lv in asked if s == skill] == [1, 2, 2, 3]
    assert not has_next(state)
    state.scores["py"].n = 12     # the stop rule counts graded questions
    assert check_stop(state) == "fixed_length"


def test_fixed_mode_follows_skill_order():
    state = make_state(grid_bank(), "B1", skill_order=("ds", "py", "sql"))
    assert [ask(state)["skill"] for _ in range(3)] == ["ds", "py", "sql"]


def test_fixed_mode_skips_an_exhausted_skill():
    bank = [x for x in grid_bank(per_cell=4) if x["skill"] != "sql"]
    state = make_state(bank, "B1")
    skills = []
    while has_next(state):
        skills.append(ask(state)["skill"])
    assert skills.count("sql") == 0 and skills.count("py") == skills.count("ds") == 4


# ---- stopping -------------------------------------------------------------------------

def make_certain(state, skill, n=6, score=1):
    for _ in range(n):
        state.scores[skill].update(2, score)


def test_uncertainty_stop_needs_every_skill_confident_with_two_questions():
    state = make_state(grid_bank())
    assert check_stop(state) is None
    for skill in ("py", "sql"):
        make_certain(state, skill)
    assert check_stop(state) is None      # ds is still unmeasured
    make_certain(state, "ds", n=1)
    assert check_stop(state) is None      # one question is not enough, whatever the SE
    make_certain(state, "ds", n=5)
    assert check_stop(state) == "confident"


def test_two_questions_are_required_even_when_se_is_small():
    state = make_state(grid_bank(), uncertainty_threshold=5.0)
    for skill in ("py", "sql", "ds"):
        state.scores[skill].update(2, 1)
    assert check_stop(state) is None
    for skill in ("py", "sql", "ds"):
        state.scores[skill].update(2, 1)
    assert check_stop(state) == "confident"


def test_max_questions_stops_the_interview():
    state = make_state(grid_bank(), max_questions=15)
    for i in range(14):
        state.scores["py"].update(2, 1)
    assert check_stop(state) is None
    state.scores["sql"].update(2, 1)
    assert check_stop(state) == "max_questions"


def test_fixed_count_stops_at_exactly_that_many():
    state = make_state(grid_bank(per_cell=5), "A1")
    for _ in range(11):
        state.scores["py"].update(2, 1)
    assert check_stop(state) is None
    state.scores["py"].update(2, 1)
    assert check_stop(state) == "fixed_length"


def test_stops_when_every_skill_is_exhausted():
    state = make_state([])
    assert check_stop(state) == "bank_exhausted"


def test_an_exhausted_skill_does_not_keep_the_others_asking():
    state = make_state(grid_bank())
    state.exhausted.add("ds")
    make_certain(state, "py")
    make_certain(state, "sql")
    assert check_stop(state) == "bank_exhausted"     # ds was never measured: say so


def test_decision_stop_rule_waits_for_a_clear_answer():
    state = make_state(grid_bank(), stop_rule="decision")
    for skill in ("py", "sql", "ds"):
        state.scores[skill].update(2, 2)
        state.scores[skill].update(2, 2)
        state.scores[skill].update(3, 2)
    assert check_stop(state) == "confident"      # clearly above the minimum of 2
    state = make_state(grid_bank(), stop_rule="decision")
    for skill in ("py", "sql", "ds"):
        state.scores[skill].update(2, 1)
        state.scores[skill].update(2, 1)
    assert check_stop(state) is None             # around the cut: undecided
