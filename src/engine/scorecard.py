"""Scoring model (spec section 2): Partial Credit Model + EAP estimate on a grid.

One `SkillScore` per skill. It keeps the log-posterior over a grid of ability values
(theta, in logits) and updates it after every graded question.
"""

from __future__ import annotations

import math

import numpy as np

# Question difficulty by level (logits). Check on the pilot.
DIFFICULTY = {1: -1.0, 2: 0.0, 3: 1.0}
STEP_OFFSET = 0.5   # step thresholds of a question of difficulty b: b - 0.5 and b + 0.5

# EAP grid and prior
GRID = np.round(np.arange(-4.0, 4.0 + 1e-9, 0.05), 2)
PRIOR_MEAN = 0.0
PRIOR_SD = 1.0

# Estimate -> level: lower edge of levels 1, 2 and 3
LEVEL_CUTS = (-1.5, -0.5, 0.5)
# Cut for a job minimum m (the theta at which a candidate reaches level m)
MINIMUM_CUT = {0: -math.inf, 1: -1.5, 2: -0.5, 3: 0.5}


def pcm_probs(theta, difficulty: float) -> np.ndarray:
    """P(score = 0, 1, 2) for a question of the given difficulty.

    `theta` may be a number or an array; the result has shape (3,) or (len(theta), 3).
    The chance of scoring x is proportional to exp(sum over steps k <= x of (theta - step k)).
    """
    theta = np.asarray(theta, dtype=float)
    steps = (difficulty - STEP_OFFSET, difficulty + STEP_OFFSET)
    logits = np.stack([np.zeros_like(theta),
                       theta - steps[0],
                       (theta - steps[0]) + (theta - steps[1])], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)   # for numerical stability
    weights = np.exp(logits)
    return weights / weights.sum(axis=-1, keepdims=True)


def level_of(theta: float) -> int:
    """Map an ability estimate to a level 0-3."""
    return sum(theta >= cut for cut in LEVEL_CUTS)


def nearest_level(theta: float) -> int:
    """Question level whose difficulty is nearest theta (a tie goes to the lower level)."""
    return min(DIFFICULTY, key=lambda level: (abs(DIFFICULTY[level] - theta), level))


class SkillScore:
    """Posterior over ability for one skill."""

    def __init__(self, start_level: int = 2):
        # Prior: Normal(mean 0, SD 1); mean 0 is level 2. A different start level
        # moves the mean to that level's difficulty.
        mean = PRIOR_MEAN + DIFFICULTY.get(start_level, 0.0)
        self.log_post = -0.5 * ((GRID - mean) / PRIOR_SD) ** 2
        self.n = 0   # graded main questions

    # ---- update ------------------------------------------------------------
    def update(self, question_level: int, score: int, weight: float = 1.0) -> None:
        """Multiply in the likelihood of `score`, raised to the evidence weight."""
        probs = pcm_probs(GRID, DIFFICULTY[question_level])[:, score]
        self.log_post = self.log_post + weight * np.log(probs)
        self.n += 1

    # ---- posterior summaries -------------------------------------------------
    def posterior(self) -> np.ndarray:
        weights = np.exp(self.log_post - self.log_post.max())
        return weights / weights.sum()

    @property
    def estimate(self) -> float:
        """Posterior mean."""
        return float(np.sum(GRID * self.posterior()))

    @property
    def se(self) -> float:
        """Posterior standard deviation."""
        post = self.posterior()
        mean = np.sum(GRID * post)
        return float(math.sqrt(np.sum(post * (GRID - mean) ** 2)))

    @property
    def level(self) -> int:
        return level_of(self.estimate)

    def prob_at_least(self, cut: float) -> float:
        """P(theta >= cut)."""
        if cut == -math.inf:
            return 1.0
        return float(self.posterior()[GRID >= cut - 1e-9].sum())

    def state(self) -> dict:
        """Snapshot for the log (matches the schema's skill_state)."""
        return {"estimate": round(self.estimate, 4), "se": round(self.se, 4),
                "level": self.level, "n": self.n}
