"""Helpers for the engine tests: real bank + fake embedder, small candidates, spies."""

import functools
import json
import random
from pathlib import Path

from jsonschema import Draft202012Validator

from conftest import fake_embed
from engine.run import run_interview
from part1.bank import load_bank
from part1.index import QuestionIndex
from shared.interfaces import SYSTEMS, Components
from shared.stubs import RuleBasedRespondent, TagDiagnoser, TagGrader

ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA = json.loads((ROOT / "schemas" / "interview_result.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(RESULT_SCHEMA)
EXAMPLE_CANDIDATE = json.loads((ROOT / "examples" / "candidate.example.json").read_text(encoding="utf-8"))
TIMESTAMPS = ("started_at", "finished_at")


@functools.lru_cache(maxsize=1)
def real_bank():
    return load_bank()


@functools.lru_cache(maxsize=1)
def real_index():
    return QuestionIndex.build(real_bank(), embed_fn=fake_embed)


def assert_valid(result):
    errors = [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in VALIDATOR.iter_errors(result)]
    assert not errors, errors


def without_timestamps(result):
    return {k: v for k, v in result.items() if k not in TIMESTAMPS}


# --------------------------------------------------------------------------
# Candidates (hand-made, built from the example key)
# --------------------------------------------------------------------------

def make_candidate(level=2, misconceptions=(), job_id="data-analyst", **changes):
    """Example candidate with every topic at `level` (an int, or a dict of topic -> level
    on top of level 2), no slips or misreads, and the given planted misconceptions."""
    topic_levels = dict.fromkeys(EXAMPLE_CANDIDATE["topic_levels"], level if isinstance(level, int) else 2)
    if isinstance(level, dict):
        topic_levels.update(level)
    candidate = {**EXAMPLE_CANDIDATE, "topic_levels": topic_levels, "misconceptions": list(misconceptions),
                 "misread_rate": 0.0, "slip_rate": 0.0, "safety_attack": None, "job_id": job_id}
    candidate.update(changes)
    return candidate


STRONG = make_candidate(3, id="CAND-STRONG")
WEAK = make_candidate(0, id="CAND-WEAK")


# --------------------------------------------------------------------------
# Question sources and other components
# --------------------------------------------------------------------------

class FilteredSource:
    """The real index with some questions hidden. Results come back in reverse id order,
    so a test fails if the engine relies on the index's own order."""

    def __init__(self, keep=lambda q: True, inner=None):
        self.inner, self.keep = inner or real_index(), keep

    def search(self, query=None, skill=None, levels=None, prefer_topics=None,
               exclude_ids=frozenset(), k=5):
        found = self.inner.search(query, skill, levels, prefer_topics, exclude_ids, k=100_000)
        return [(q, s) for q, s in reversed(found) if self.keep(q)][:k]


class CountingDiagnoser:
    def __init__(self, inner=None):
        self.inner, self.calls = inner or TagDiagnoser(), 0

    def diagnose(self, question, answer, history=()):
        self.calls += 1
        return self.inner.diagnose(question, answer, history)


class FakeGenerator:
    """Returns a simple generated question on the requested topic."""

    def __init__(self):
        self.calls = []

    def generate(self, skill, level, topic):
        self.calls.append((skill, level, topic))
        return {"id": f"GEN-{len(self.calls)}", "skill": skill, "level": level, "topics": [topic],
                "question": f"Generated question about {topic}.", "model_answer": "-", "key_points": ["-"],
                "rubric": {"0": "-", "1": "-", "2": "-"}, "misconceptions": [], "source": "generated-live"}


def make_components(source=None, grader=None, diagnoser=None, **extra):
    return Components(question_source=source or real_index(), grader=grader or TagGrader(),
                      diagnoser=diagnoser or TagDiagnoser(),
                      misconceptions=real_bank().misconception_by_id, **extra)


def job_for(candidate, **changes):
    return {**real_bank().get_job(candidate["job_id"]), **changes}


def run(candidate, system="A2", *, seed=0, job=None, components=None, respondent=None, **config_changes):
    """One interview of `candidate` under a preset, with the stubs. Returns (result, respondent)."""
    config = SYSTEMS[system].with_(seed=seed, **config_changes)
    respondent = respondent or RuleBasedRespondent(candidate)
    result = run_interview(job or job_for(candidate), respondent, config, components or make_components(),
                           candidate_id=candidate["id"], rng=random.Random(seed))
    return result, respondent


def misconception_setup(misconception_id, skill_level=3):
    """A candidate holding one misconception, a job that requires only its topic, and a source in
    which every question on that topic lists the misconception, so it is certain to be met."""
    record = real_bank().get_misconception(misconception_id)
    topic = record["topic"]
    candidate = make_candidate(skill_level, [misconception_id], id="CAND-MISC")
    job = job_for(candidate, required_topics=[topic], min_levels={"py": 1, "sql": 1, "ds": 1})

    def keep(q):
        return topic not in q["topics"] or misconception_id in q["misconceptions"]

    return candidate, job, FilteredSource(keep), topic
