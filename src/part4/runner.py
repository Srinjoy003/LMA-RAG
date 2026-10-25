"""The experiment runner: every chosen system interviews every candidate (docs/part4.md).

One result file per interview, `results/<run>/<system>/<job_id>/<candidate_id>.json`, with the
candidate's plans merged in as `truth` fields and the ground truth at the top level. Files that
exist are skipped, so a run stopped by a quota limit continues where it left off. An interview
that ends in an error is not saved, so the next run tries it again.

The grader and the simulator each run on a provider from llm.providers, or on the stubs
("stub": tag-reading grader and template answers, no LLM).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from engine.factory import build_components
from engine.run import ROOT, run_interview
from shared.interfaces import SYSTEMS
from shared.stubs import RuleBasedRespondent

from .candidates import truth_for
from .respondent import SimulatedRespondent
from .simulator import LLMCandidateSimulator

RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Setup:
    grader: str = "stub"             # stub, or a provider name: gemini, groq, ollama
    simulator: str = "stub"
    simulator_model: str | None = None

    def problems(self) -> list[str]:
        """Combinations that cannot give meaningful results."""
        if (self.grader == "stub") != (self.simulator == "stub"):
            return ["the stub grader only reads the stub simulator's hidden tags: use stub for both or neither"]
        return []

    def warnings(self) -> list[str]:
        if self.grader != "stub" and self.grader == self.simulator:
            return [f"grader and simulator both use {self.grader}: the simulator should be a different model family"]
        return []


def merge_truths(result: dict, truths: list) -> None:
    """Add the respondent's hidden plans to the result as `truth` fields on turns and follow-ups."""
    follow_ups_seen: dict[int, int] = {}
    for turn_no, kind, plan in truths:
        if turn_no > len(result["turns"]):
            continue    # a reply for a question that never made it into the log
        turn = result["turns"][turn_no - 1]
        if kind == "main":
            turn["truth"] = dict(plan)
        else:
            i = follow_ups_seen.get(turn_no, 0)
            follow_ups_seen[turn_no] = i + 1
            if i < len(turn["follow_ups"]):
                turn["follow_ups"][i]["truth"] = dict(plan)


def make_simulator(setup: Setup):
    """The LLM simulator for the setup, or None for the stub (template answers with tags)."""
    if setup.simulator == "stub":
        return None
    from llm.providers import PROVIDERS, provider_client, provider_model
    model = setup.simulator_model or provider_model(setup.simulator) or "gemini-3.5-flash-lite"
    return LLMCandidateSimulator(provider_client(setup.simulator), model,
                                 reasoning_effort=PROVIDERS[setup.simulator].reasoning_effort)


def run_one(system: str, candidate: dict, job: dict, setup: Setup, simulator, root: Path = ROOT) -> dict:
    """One interview with truth merged in."""
    config = SYSTEMS[system]
    if setup.grader == "stub":
        components = build_components(config, live=False, root=root)
    else:
        from llm.providers import apply_provider
        config = apply_provider(config, setup.grader)
        components = build_components(config, live=True, root=root, provider=setup.grader)
    respondent = (RuleBasedRespondent(candidate) if simulator is None
                  else SimulatedRespondent(candidate, simulator, components.misconceptions))
    result = run_interview(job, respondent, config, components, candidate_id=candidate["id"],
                           rng=random.Random(config.seed))
    merge_truths(result, respondent.truths)
    result["truth"] = truth_for(candidate, job)
    result["simulator"] = {"provider": setup.simulator, "model": getattr(simulator, "model", "stub")}
    return result


def result_path(run_dir: Path, system: str, job_id: str, candidate_id: str) -> Path:
    return run_dir / system / job_id / f"{candidate_id}.json"


def run_experiment(systems: list[str], candidates: list[dict], jobs: dict[str, dict], setup: Setup,
                   run_dir: Path, *, rerun: bool = False, root: Path = ROOT, report=print) -> dict:
    """Run every (system, candidate) pair that has no result yet. Returns counts of
    done / skipped / failed; stops early on a quota error (rerun later to continue)."""
    problems = setup.problems()
    if problems:
        raise ValueError("; ".join(problems))
    for warning in setup.warnings():
        report("Warning: " + warning)
    simulator = make_simulator(setup)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for system in systems:
        for candidate in candidates:
            path = result_path(run_dir, system, candidate["job_id"], candidate["id"])
            if path.exists() and not rerun:
                counts["skipped"] += 1
                continue
            result = run_one(system, candidate, jobs[candidate["job_id"]], setup, simulator, root)
            if result["stop_reason"] == "error":
                counts["failed"] += 1
                report(f"{system} {candidate['id']}: failed, not saved ({'; '.join(result['errors'])[:200]})")
                if any("Quota" in e for e in result["errors"]):
                    report("Stopped: the provider's quota is used up. Run again later to continue.")
                    return counts
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            counts["done"] += 1
            report(f"{system} {candidate['id']}: {result['report']['decision']}, {result['exchanges']} exchanges")
    return counts
