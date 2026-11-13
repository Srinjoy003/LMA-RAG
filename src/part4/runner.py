"""The experiment runner: every chosen system interviews every candidate (docs/part4.md).

One result file per interview, `results/<run>/<system>/<job_id>/<candidate_id>.json`, with the
candidate's plans merged in as `truth` fields and the ground truth at the top level. Files that
exist are skipped, so a run stopped by a quota limit continues where it left off. An interview
that ends in an error is not saved but logged in `failures.jsonl`, so the next run tries it
again; after MAX_ATTEMPTS logged failures the pair is excluded (protocol section 4).

Each run folder has a manifest (part4/manifest.py). Before any interview the runner checks
that the code, bank, prompts, setup, each condition's configuration and each candidate file
match it, and refuses otherwise, so an old or different run is never silently extended.
Every saved result names its run and the manifest fingerprint.

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

from .candidates import candidate_digest, truth_for
from .manifest import RunManifest, failure_counts, git_state, record_failure
from .respondent import SimulatedRespondent
from .simulator import LLMCandidateSimulator

RESULTS_DIR = ROOT / "results"
CODE_ROOT = ROOT          # the engine source that is hashed into every manifest
MAX_ATTEMPTS = 3          # failed attempts before a (system, candidate) pair is excluded


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


def condition_config(system: str, setup: Setup):
    """The configuration a condition actually runs with (live providers set the model names)."""
    config = SYSTEMS[system]
    if setup.grader != "stub":
        from llm.providers import apply_provider
        config = apply_provider(config, setup.grader)
    return config


def setup_record(setup: Setup, simulator) -> dict:
    return {"grader": setup.grader, "simulator": setup.simulator,
            "simulator_model": getattr(simulator, "model", None) if simulator is not None else "stub"}


def run_one(system: str, candidate: dict, job: dict, setup: Setup, simulator, root: Path = ROOT) -> dict:
    """One interview with truth merged in."""
    config = condition_config(system, setup)
    if setup.grader == "stub":
        components = build_components(config, live=False, root=root)
    else:
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
                   run_dir: Path, *, rerun: bool = False, root: Path = ROOT, report=print,
                   operator: str | None = None, share: str | None = None,
                   code_root: Path | None = None) -> dict:
    """Run every (system, candidate) pair that has no result yet. Returns counts of
    done / skipped / failed / excluded; stops early on a quota error (rerun later to continue).

    Raises part4.manifest.RunManifestError, before any interview, if `run_dir` was made with
    different code, bank, prompts, setup, condition configurations or candidate files."""
    problems = setup.problems()
    if problems:
        raise ValueError("; ".join(problems))
    for warning in setup.warnings():
        report("Warning: " + warning)
    simulator = make_simulator(setup)
    run_dir = Path(run_dir)
    code_root = Path(code_root or CODE_ROOT)
    manifest = RunManifest.open(run_dir, code_root=code_root, bank_root=root, setup=setup_record(setup, simulator))
    for system in systems:
        manifest.check_condition(system, json.loads(json.dumps(condition_config(system, setup).to_dict())))
    manifest.check_candidates(candidates, {c["id"]: candidate_digest(c) for c in candidates})
    manifest.add_session(operator=operator, share=share, systems=systems,
                         candidate_ids=[c["id"] for c in candidates], git=git_state(code_root))
    manifest.save()
    tag = {"name": run_dir.name, "fingerprint": manifest.fingerprint_id}

    failures = failure_counts(run_dir)
    counts = {"done": 0, "skipped": 0, "failed": 0, "excluded": 0}
    for system in systems:
        for candidate in candidates:
            path = result_path(run_dir, system, candidate["job_id"], candidate["id"])
            if path.exists() and not rerun:
                counts["skipped"] += 1
                continue
            if failures.get((system, candidate["id"]), 0) >= MAX_ATTEMPTS and not path.exists():
                counts["excluded"] += 1
                report(f"{system} {candidate['id']}: excluded after {MAX_ATTEMPTS} failed attempts")
                continue
            result = run_one(system, candidate, jobs[candidate["job_id"]], setup, simulator, root)
            if result["stop_reason"] == "error":
                quota = any("Quota" in e for e in result["errors"])
                counts["failed"] += 1
                report(f"{system} {candidate['id']}: failed, not saved ({'; '.join(result['errors'])[:200]})")
                if quota:       # a quota stop is a resume, not a failed attempt
                    report("Stopped: the provider's quota is used up. Run again later to continue.")
                    return counts
                record_failure(run_dir, system, candidate["id"], result["errors"])
                failures[(system, candidate["id"])] = failures.get((system, candidate["id"]), 0) + 1
                continue
            result["run"] = dict(tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            counts["done"] += 1
            report(f"{system} {candidate['id']}: {result['report']['decision']}, {result['exchanges']} exchanges")
    return counts
