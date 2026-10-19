import math

import numpy as np
import pytest

from engine.scorecard import (DIFFICULTY, GRID, MINIMUM_CUT, SkillScore, level_of, nearest_level,
                              pcm_probs)


@pytest.mark.parametrize("difficulty", [-1.0, 0.0, 1.0])
def test_pcm_probabilities_sum_to_one(difficulty):
    theta = np.linspace(-4, 4, 17)
    probs = pcm_probs(theta, difficulty)
    assert probs.shape == (17, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()
    assert np.isclose(pcm_probs(0.3, difficulty).sum(), 1.0)


def test_pcm_matches_the_spec_formula():
    # a question of difficulty b has steps b - 0.5 and b + 0.5; P(x) ~ exp(sum of (theta - step))
    theta, b = 0.7, 1.0
    s1, s2 = b - 0.5, b + 0.5
    raw = np.array([1.0, math.exp(theta - s1), math.exp((theta - s1) + (theta - s2))])
    assert np.allclose(pcm_probs(theta, b), raw / raw.sum())


def test_higher_ability_means_a_higher_expected_score():
    expected = [pcm_probs(t, 0.0) @ np.array([0, 1, 2]) for t in (-3, -1, 0, 1, 3)]
    assert expected == sorted(expected)


def test_harder_questions_score_lower():
    expected = [pcm_probs(0.0, DIFFICULTY[level]) @ np.array([0, 1, 2]) for level in (1, 2, 3)]
    assert expected[0] > expected[1] > expected[2]


def test_grid_and_difficulties():
    assert GRID[0] == -4 and GRID[-1] == 4 and len(GRID) == 161
    assert np.allclose(np.diff(GRID), 0.05)
    assert DIFFICULTY == {1: -1.0, 2: 0.0, 3: 1.0}


def test_no_data_gives_the_prior():
    score = SkillScore()
    assert score.estimate == pytest.approx(0.0, abs=1e-9)
    assert score.se == pytest.approx(1.0, abs=0.01)    # Normal(0, 1), cut off at +-4
    assert score.level == 2 and score.n == 0


def test_start_level_moves_the_prior_mean():
    assert SkillScore(start_level=3).estimate == pytest.approx(1.0, abs=0.01)
    assert SkillScore(start_level=1).estimate == pytest.approx(-1.0, abs=0.01)


def test_all_correct_gives_a_finite_high_estimate():
    score = SkillScore()
    for _ in range(10):
        score.update(3, 2)
    assert math.isfinite(score.estimate) and math.isfinite(score.se)
    assert 1.0 < score.estimate < 4.0
    assert score.level == 3


def test_all_wrong_gives_a_finite_low_estimate():
    score = SkillScore()
    for _ in range(10):
        score.update(1, 0)
    assert -4.0 < score.estimate < -1.5
    assert score.level == 0


def test_se_shrinks_as_answers_are_added():
    score = SkillScore()
    ses = [score.se]
    for s in (2, 1, 2, 0, 1, 2):
        score.update(2, s)
        ses.append(score.se)
    assert all(a > b for a, b in zip(ses, ses[1:]))
    assert score.n == 6


def test_a_lower_evidence_weight_moves_the_estimate_less():
    moves = {}
    for weight in (0.0, 0.5, 0.67, 1.0):
        score = SkillScore()
        score.update(2, 2, weight)
        moves[weight] = score.estimate
    assert moves[0.0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < moves[0.5] < moves[0.67] < moves[1.0]


def test_a_lower_weight_also_shrinks_se_less():
    full, half = SkillScore(), SkillScore()
    full.update(2, 1, 1.0)
    half.update(2, 1, 0.5)
    assert full.se < half.se < SkillScore().se


def test_harder_questions_answered_correctly_count_for_more():
    easy, hard = SkillScore(), SkillScore()
    easy.update(1, 2)
    hard.update(3, 2)
    assert hard.estimate > easy.estimate


@pytest.mark.parametrize("theta,level", [(-4, 0), (-1.51, 0), (-1.5, 1), (-0.51, 1), (-0.5, 2),
                                         (0.49, 2), (0.5, 3), (4, 3)])
def test_estimate_to_level(theta, level):
    assert level_of(theta) == level


@pytest.mark.parametrize("theta,level", [(-3, 1), (-0.6, 1), (-0.4, 2), (0.0, 2), (0.4, 2), (0.6, 3), (3, 3)])
def test_nearest_question_level(theta, level):
    assert nearest_level(theta) == level


def test_nearest_level_tie_goes_to_the_lower_level():
    assert nearest_level(-0.5) == 1 and nearest_level(0.5) == 2


def test_probability_of_meeting_a_minimum():
    prior = SkillScore()
    assert prior.prob_at_least(MINIMUM_CUT[0]) == 1.0
    assert prior.prob_at_least(MINIMUM_CUT[2]) == pytest.approx(0.69, abs=0.02)   # P(Z >= -0.5)
    assert prior.prob_at_least(MINIMUM_CUT[3]) == pytest.approx(0.31, abs=0.02)   # P(Z >= 0.5)
    cuts = [prior.prob_at_least(MINIMUM_CUT[m]) for m in (0, 1, 2, 3)]
    assert cuts == sorted(cuts, reverse=True)


def test_state_snapshot_has_the_schema_fields():
    score = SkillScore()
    score.update(2, 2)
    assert set(score.state()) == {"estimate", "se", "level", "n"}
    assert score.state()["n"] == 1
