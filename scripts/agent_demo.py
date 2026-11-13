r"""Run one interview with the tool-calling agent driver and show what the agent did (Step 7).

    python scripts/agent_demo.py                                   # offline: scripted agent, stubs
    python scripts/agent_demo.py --probe-rules                     # plus deliberate rule-breaking calls
    python scripts/agent_demo.py --system A2EV --candidate CAND-002 --probe-rules   # with verify_sql_result
    python scripts/agent_demo.py --grader gemini --simulator groq --simulator-model openai/gpt-oss-120b
                                                                   # live: the provider's model is the agent

Prints every tool call (accepted, refused or invalid, with the reason and its time), the agent's
summary with a consistency check against the computed outcome, and, with --compare-code (on by
default), the same interview through the code driver: same questions, same decision, and the
extra LLM calls and time the agent costs. --out writes everything as JSON (never overwritten).

Offline, the agent is a scripted policy (engine/demo.py) and grading reads the stub simulator's
tags, so the demo shows the rules being enforced, not model quality.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.demo import ScriptedAgent, compare_drivers, summary_check  # noqa: E402
from engine.factory import build_components  # noqa: E402
from engine.run import run_interview  # noqa: E402
from part1.bank import load_bank  # noqa: E402
from part4.candidates import CANDIDATES_DIR, load_candidates, truth_for  # noqa: E402
from part4.respondent import SimulatedRespondent  # noqa: E402
from part4.runner import Setup, make_simulator  # noqa: E402
from shared.interfaces import SYSTEMS  # noqa: E402
from shared.stubs import RuleBasedRespondent  # noqa: E402

PROVIDERS = ["stub", "gemini", "groq", "ollama"]


def build(config, setup: Setup, probes: bool):
    """Components and a fresh-respondent factory; the agent's LLM is scripted offline."""
    if setup.grader == "stub":
        components = build_components(config, live=False, root=ROOT)
        components.llm = ScriptedAgent(probes=probes)
        return components, None
    components = build_components(config, live=True, root=ROOT, provider=setup.grader)
    return components, make_simulator(setup)


def interview(system: str, driver: str, candidate: dict, job: dict, setup: Setup, probes: bool) -> tuple[dict, float]:
    config = SYSTEMS[system].with_(driver=driver)
    if setup.grader != "stub":
        from llm.providers import apply_provider
        config = apply_provider(config, setup.grader)
    components, simulator = build(config, setup, probes and driver == "agent")
    respondent = (RuleBasedRespondent(candidate) if simulator is None
                  else SimulatedRespondent(candidate, simulator, components.misconceptions))
    started = time.perf_counter()
    result = run_interview(job, respondent, config, components, candidate_id=candidate["id"],
                           rng=random.Random(config.seed))
    result["truth"] = truth_for(candidate, job)
    return result, time.perf_counter() - started


def show_trace(result: dict) -> None:
    """Every tool call in order, with its outcome and time; verification shows status only."""
    verified = {t["index"]: t["verification"] for t in result["turns"] if t.get("verification")}
    print(f"{'turn':>4}  {'tool':<24} {'result':<8} {'ms':>8}  note")
    for call in (result.get("agent") or {}).get("tool_timings") or []:
        error = call.get("error") or ""
        status = "ok" if call["ok"] else ("invalid" if error.startswith("invalid") else "REFUSED")
        note = error
        if call["name"] == "verify_sql_result" and call["ok"] and call["turn"] in verified:
            v = verified[call["turn"]]
            note = f"status {v['status']}, {v.get('row_count')} rows (the rows go to the grader only)"
        print(f"{call['turn']:>4}  {call['name']:<24} {status:<8} {call['ms']:>8.2f}  {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="A2E", choices=sorted(SYSTEMS))
    parser.add_argument("--candidate", default="CAND-001")
    parser.add_argument("--candidates-dir", type=Path, default=CANDIDATES_DIR)
    parser.add_argument("--grader", default="stub", choices=PROVIDERS,
                        help="stub (offline, scripted agent) or a provider whose model also runs the agent")
    parser.add_argument("--simulator", default="stub", choices=PROVIDERS)
    parser.add_argument("--simulator-model")
    parser.add_argument("--probe-rules", action="store_true", help="scripted agent only: add rule-breaking calls")
    parser.add_argument("--no-compare-code", dest="compare_code", action="store_false")
    parser.add_argument("--out", type=Path, help="write the full demo record as JSON (refuses to overwrite)")
    args = parser.parse_args()

    setup = Setup(grader=args.grader, simulator=args.simulator, simulator_model=args.simulator_model)
    if setup.problems():
        parser.error("; ".join(setup.problems()))
    if args.probe_rules and setup.grader != "stub":
        parser.error("--probe-rules drives the scripted agent; a live model makes its own calls")
    if args.out and args.out.exists():
        parser.error(f"{args.out} exists; choose a new --out")
    try:
        candidate = load_candidates(args.candidates_dir, [args.candidate])[0]
    except KeyError as e:
        parser.error(str(e).strip("'\""))
    job = {j["id"]: j for j in load_bank(ROOT).jobs}[candidate["job_id"]]

    agent, agent_seconds = interview(args.system, "agent", candidate, job, setup, args.probe_rules)
    info = agent.get("agent") or {}
    print(f"Agent-driven interview: system {args.system}, candidate {candidate['id']} ({job['id']}), "
          f"agent {'scripted (offline, no LLM)' if setup.grader == 'stub' else info.get('model')}\n")
    show_trace(agent)
    print(f"\nSteps {info.get('steps')}, tool calls {info.get('tool_calls')}, refused {info.get('refused_calls')}, "
          f"invalid {info.get('invalid_calls')}; stop reason {agent['stop_reason']}"
          + (f"; errors: {'; '.join(agent['errors'])}" if agent["errors"] else ""))
    verified = [t["verification"] for t in agent["turns"] if t.get("verification")]
    if SYSTEMS[args.system].sql_verification:
        statuses = ", ".join(f"{v['question_id']} {v['status']}" for v in verified) or "no SQL read-code question asked"
        print(f"SQL verification: {statuses}")
    report = agent["report"]
    check = summary_check(report)
    print(f"\nDecision (computed by code): {report['decision']}. Levels: "
          + ", ".join(f"{s} {v['level']}" for s, v in report["levels"].items()))
    print(f"Agent summary: {report.get('summary')}")
    print("Summary consistency (heuristic): " + ("consistent" if check["consistent"]
                                                 else "PROBLEMS: " + "; ".join(check["problems"])))

    record = {"system": args.system, "candidate": candidate["id"], "setup": vars(setup), "agent_result": agent,
              "agent_seconds": round(agent_seconds, 3), "summary_check": check}
    if args.compare_code:
        code, code_seconds = interview(args.system, "code", candidate, job, setup, False)
        comparison = compare_drivers(agent, code)
        comparison.update(agent_seconds=round(agent_seconds, 3), code_seconds=round(code_seconds, 3))
        record.update(code_result=code, comparison=comparison)
        print(f"\nCode driver, same setup: same questions {comparison['same_questions']}, same decision "
              f"{comparison['same_decision']}, same levels {comparison['same_levels']}; exchanges "
              f"{comparison['exchanges']['agent']} vs {comparison['exchanges']['code']}")
        print(f"Agent overhead: {comparison['agent_llm_calls']} agent steps (each a model call when live); wall time "
              f"{agent_seconds:.2f} s (agent) vs {code_seconds:.2f} s (code)"
              + ("" if setup.grader != "stub" else " (offline: the scripted agent costs no calls)"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
