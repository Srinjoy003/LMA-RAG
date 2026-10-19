"""The code driver: runs one interview and returns the result (spec sections 1, 3, 7).

`run_interview` is a thin loop over `InterviewSession` (engine/session.py): it hands each
prompt to the respondent and submits the reply. With `driver="agent"` it runs the
LLM-agent driver (engine/agent.py) instead.

The engine only sees reply text. It never reads candidate keys or `respondent.truths`;
the experiment runner adds `truth` fields to the result afterwards.
"""

from __future__ import annotations

import random

from .result import ROOT, TOPIC_ID, required_topics_for, save_result   # noqa: F401  (re-exported)
from .session import InterviewSession


def run_interview(job: dict, respondent, config, components, *, candidate_id: str | None = None,
                  rng: random.Random | None = None) -> dict:
    """Run one interview and return the result dict."""
    # A metered LLM client (engine.factory.MeteredLLM) lets the result report every LLM call
    # the interview made: grader, diagnoser, topic extractor and agent.
    totals = getattr(getattr(components, "llm", None), "usage_totals", None)
    before = totals() if totals else None

    if config.driver == "agent":
        from .agent import run_agent_interview      # imported here: the agent driver needs an LLM
        result = run_agent_interview(job, respondent, config, components, candidate_id=candidate_id, rng=rng)
    else:
        session = InterviewSession(job, config, components, candidate_id=candidate_id, rng=rng)
        while (prompt := session.next_prompt()) is not None:
            try:
                answer = respondent.reply(prompt["text"], question=prompt["question"],
                                          kind=prompt["kind"], turn=prompt["turn"])
            except Exception as e:      # the respondent failed: end the interview with what was scored
                session.fail(e)
                break
            session.submit(answer)
        result = session.result()

    if totals:
        from .factory import usage_delta
        result["usage"] = usage_delta(before, totals())
    return result
