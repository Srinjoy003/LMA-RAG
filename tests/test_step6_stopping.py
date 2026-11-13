"""Level-band uncertainty, boundary stopping, and control/version contracts."""

from dataclasses import fields

import numpy as np
import pytest

from engine.scorecard import GRID, LEVEL_CUTS, SkillScore, level_of
from engine.selection import check_stop
from shared.interfaces import SYSTEMS
from engine_support import STRONG, assert_valid, run
from test_engine_selection import make_state, grid_bank


def gaussian(mean, sd=0.15, n=3):
    score = SkillScore()
    score.log_post = -0.5 * ((GRID - mean) / sd) ** 2
    score.n = n
    return score


def test_band_masses_partition_the_posterior_and_are_reported_for_eap_level():
    score = SkillScore()
    probabilities = score.level_probabilities()
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert score.level_probability == probabilities[score.level]
    assert score.uncertainty()["level_probability"] == round(score.level_probability, 6)


@pytest.mark.parametrize("cut", LEVEL_CUTS)
def test_exact_cut_grid_point_belongs_only_to_upper_band(cut):
    score = SkillScore()
    score.log_post[:] = -np.inf
    score.log_post[np.where(GRID == cut)[0][0]] = 0
    assert score.level_probabilities()[level_of(cut)] == 1
    assert sum(score.level_probabilities().values()) == 1


def test_eap_band_is_not_replaced_by_the_most_likely_band():
    score = SkillScore()
    score.log_post[:] = -np.inf
    score.log_post[np.where(GRID == -2)[0][0]] = np.log(.55)
    score.log_post[np.where(GRID == 2)[0][0]] = np.log(.45)
    assert max(score.level_probabilities(), key=score.level_probabilities().get) == 0
    assert score.level != 0
    assert score.level_probability == 0


def test_small_se_at_a_cut_does_not_satisfy_new_rule():
    near = gaussian(-.25)
    assert near.se < .6 and near.level_probability < .7
    assert not near.meets_level_confidence(.6, .7)
    assert gaussian(.25).meets_level_confidence(.6, .7)
    assert not gaussian(.25, n=1).meets_level_confidence(.6, .7)
    assert not gaussian(.25, sd=.8).meets_level_confidence(.6, .7)


def test_legacy_stops_at_boundary_but_new_rule_waits_and_respects_cap():
    state = make_state(grid_bank())
    state.scores = {s: gaussian(-.25) for s in state.scores}
    assert check_stop(state) == "confident"
    state.config = SYSTEMS["A2EB"]
    assert check_stop(state) is None
    state.config = state.config.with_(max_questions=9)
    assert check_stop(state) == "max_questions"


def test_bank_exhaustion_is_not_reported_as_confident_at_boundary():
    state = make_state(grid_bank())
    state.config = SYSTEMS["A2EB"]
    state.scores = {s: gaussian(-.25) for s in state.scores}
    state.exhausted.update(state.scores)
    assert check_stop(state) == "bank_exhausted"


def test_all_skills_must_meet_both_requirements():
    state = make_state(grid_bank())
    state.config = SYSTEMS["A2EB"]
    state.scores = {s: gaussian(.25) for s in state.scores}
    assert check_stop(state) == "confident"
    state.scores["sql"] = gaussian(-.25)
    assert check_stop(state) is None


@pytest.mark.parametrize("threshold", [0, -1, 1.01, float("nan")])
def test_bad_probability_threshold_fails_before_interview(threshold):
    with pytest.raises(ValueError, match="level_probability_threshold"):
        make_state(grid_bank(), stop_rule="level_confidence", level_probability_threshold=threshold)


def test_boundary_presets_change_only_stopping_rule():
    for revised, reference in [("A2EB", "A2E"), ("A2B", "A2")]:
        diffs = {f.name for f in fields(SYSTEMS[reference])
                 if getattr(SYSTEMS[revised], f.name) != getattr(SYSTEMS[reference], f.name)}
        assert diffs == {"system", "stop_rule"}


@pytest.mark.parametrize("system", ["A2", "A2E", "A2B", "A2EB"])
def test_reports_expose_uncertainty_and_validate(system):
    from engine.factory import build_components
    config = SYSTEMS[system]
    result, _ = run(STRONG, system, components=build_components(config, live=False))
    assert result["errors"] == []
    for row in result["report"]["levels"].values():
        assert set(row["level_probabilities"]) == {"0", "1", "2", "3"}
        assert sum(row["level_probabilities"].values()) == pytest.approx(1, abs=3e-6)
        assert row["level_probability"] == row["level_probabilities"][str(row["level"])]
        if system in ("A2B", "A2EB") and result["stop_reason"] == "confident":
            assert row["level_probability"] >= config.level_probability_threshold
    assert_valid(result)


def test_boundary_rule_works_through_agent_safeguards():
    from engine.factory import build_components
    from test_engine_agent import ScriptedLLM, agent_run
    config = SYSTEMS["A2EB"]
    result = agent_run(STRONG, ScriptedLLM(), "A2EB", components=build_components(config, live=False))
    assert result["errors"] == []
    assert_valid(result)


def test_live_scorecard_uses_the_new_gate_and_reports_band_mass():
    from test_engine_session import new_session
    session = new_session(STRONG, "A2EB")
    session.state.scores = {s: gaussian(-.25) for s in session.state.scores}
    for row in session.scorecard_view().values():
        assert row["se"] < .6
        assert row["level_probability"] < .7
        assert not row["confident"]


def test_saved_legacy_reports_remain_schema_compatible():
    result, _ = run(STRONG)
    for row in result["report"]["levels"].values():
        row.pop("level_probability")
        row.pop("level_probabilities")
    assert_valid(result)


def test_analysis_keeps_wrong_levels_when_every_interview_hits_cap():
    from test_step5_evaluation import load_script
    result = {"config": {}, "stop_reason": "max_questions",
              "truth": {"levels": {s: 0 for s in ("py", "sql", "ds")}},
              "report": {"levels": {s: {"level": 1, "level_probability": .55}
                                     for s in ("py", "sql", "ds")}}}
    counts = load_script("check_stopping").audit([result])
    assert counts["confident_interviews"]["confident"] == 0
    assert counts["all_stops_by_true_level"]["0"] == {
        "wrong": 3, "assessed": 3, "over": 3, "under": 0, "error_rate": 1}
    assert counts["wrong_skills_below_probability_threshold"] == 3


def test_manifest_displays_agent_topic_and_followup_prompt_sources():
    from engine_support import ROOT
    from part4.manifest import fingerprint
    sources = fingerprint(code_root=ROOT, bank_root=ROOT, setup={})["prompt_sources_sha256"]
    assert {"src/engine/agent.py", "src/part1/extract_topics.py", "src/engine/followups.py"} <= set(sources)
