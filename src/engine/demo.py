"""Demonstration of the tool-calling agent driver (Step 7), offline or with a live model.

`ScriptedAgent` is an offline stand-in for the interviewer LLM: it reads the last successful
tool result and makes the next call of the documented procedure, so the demo runs with no key.
With `probes=True` it also makes one deliberately rule-breaking call before several steps (end
the interview at once, ask a question the rules did not choose, record a score before grading,
search while a question is open, grade before verification, call a tool that does not exist).
The tools must refuse each one and the interview must go on unchanged. Its summary repeats the
computed outcome, so it is consistent by construction; with a live model the same check is a
real test.

`summary_check` is a heuristic consistency check of the agent's summary against the outcome the
system computed. `compare_drivers` contrasts the agent run with the code driver on the same setup.
"""

from __future__ import annotations

import json
import re

SKILL_NAMES = {"py": "Python", "sql": "SQL", "ds": "data structures"}


class ScriptedAgent:
    """An offline `LLMClient` that drives the interview through the tools."""

    model = "scripted-agent"

    def __init__(self, probes: bool = False):
        self.probes = probes
        self.probe_counts: dict[str, int] = {}     # probes already made before each kind of step
        self.done_probes: set[str] = set()         # steps whose probes are all made
        self.n = 0

    # -- policy -----------------------------------------------------------------------------
    @staticmethod
    def summary_for(outcome: dict) -> str:
        levels = ", ".join(f"{SKILL_NAMES[s]} level {v['level']} (job minimum {v['minimum']})"
                           for s, v in outcome["levels"].items())
        rec = outcome["recommendation"]
        verdict = {"needs_review": "The evidence is not conclusive, so a human should review this candidate.",
                   "fit": "The system's recommendation is fit.",
                   "not_fit": "The system's recommendation is not fit."}.get(rec, f"Recommendation: {rec}.")
        return f"Levels: {levels}. {verdict} {outcome.get('reason', '')}".strip()

    def decide(self, last_name: str | None, last: dict | None) -> tuple[str, dict]:
        if last_name is None:
            return "search_question_bank", {}
        if last_name == "search_question_bank":
            self._question = last
            return "ask_candidate", {"question_id": last["question_id"]}
        if last_name == "ask_candidate":
            if last.get("verification_required"):
                return "verify_sql_result", {"question_id": self._question["question_id"]}
            return "grade_answer", {}
        if last_name == "verify_sql_result":
            return "grade_answer", {}
        if last_name in ("grade_answer", "ask_follow_up"):
            return ("ask_follow_up", {}) if last["follow_up_allowed"] else ("update_scorecard", {})
        if last_name == "update_scorecard":
            if last["can_end_interview"]:
                return "end_interview", {"summary": self.summary_for(last["outcome"])}
            return "search_question_bank", {}
        raise RuntimeError(f"no successful call to continue from: {last_name}")

    def probe_before(self, action: str, args: dict, last: dict | None, tool_names: set[str]):
        """Rule-breaking calls made before a kind of step is first taken, one per model turn.
        A refused call changes nothing, so the policy then picks the same step again and the next
        probe (or the real call) follows."""
        if not self.probes:
            return None
        probes = {
            "search_question_bank": [("end_interview", {"summary": "Finished early."})],
            "ask_candidate": [("ask_candidate", {"question_id": "NOT-THE-CHOSEN-QUESTION"})],
            "grade_answer": [("update_scorecard", {})],
            "update_scorecard": [("search_question_bank", {})],
            "end_interview": [("delete_candidate_record", {})],
            "verify_sql_result": [("grade_answer", {})],          # grading before verification
        }
        if action == "grade_answer" and "verify_sql_result" in tool_names and \
                not (last or {}).get("verification_required"):
            # verification of a question that is not SQL read-code
            probes["grade_answer"].append(("verify_sql_result", {"question_id": self._question["question_id"]}))
        done = self.probe_counts.get(action, 0)
        queue = probes.get(action, [])
        if action in self.done_probes or done >= len(queue):
            self.done_probes.add(action)       # only before the first real call of this step
            return None
        self.probe_counts[action] = done + 1
        return queue[done]

    # -- LLMClient --------------------------------------------------------------------------
    @staticmethod
    def last_ok(messages: list[dict]) -> tuple[str | None, dict | None]:
        names, last = {}, (None, None)
        for message in messages:
            for call in message.get("tool_calls") or []:
                names[call["id"]] = call["function"]["name"]
            if message.get("role") == "tool":
                content = json.loads(message["content"])
                if content.get("ok"):
                    last = (names[message["tool_call_id"]], content)
        return last

    def complete(self, messages, *, model, temperature=0.0, json_schema=None, max_tokens=1024,
                 tools=None, reasoning_effort=None, cache_tag=None):
        tool_names = {t["function"]["name"] for t in tools or []}
        name, args = self.decide(*self.last_ok(messages))
        probe = self.probe_before(name, args, self.last_ok(messages)[1], tool_names)
        if probe is not None:
            name, args = probe
        call_id = f"call_{self.n}"
        self.n += 1
        raw = {"role": "assistant", "content": None,
               "tool_calls": [{"id": call_id, "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}}]}
        return {"text": "", "tool_calls": [{"id": call_id, "name": name, "arguments": args}],
                "raw_message": raw, "model": self.model, "input_tokens": 0, "output_tokens": 0}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

CLAIMS = {   # phrases that state a verdict the outcome must support
    "fit": [r"\bnot (a )?fit\b", r"\breject", r"\bshould not be hired\b"],
    "not_fit": [r"\b(is|a|strong) fit\b", r"\bhire\b", r"\bshould be hired\b", r"\brecommend(ed)? (for|hiring)\b"],
    "needs_review": [r"\b(is|a|strong) fit\b", r"\bnot (a )?fit\b", r"\bhire\b", r"\breject"],
}


def summary_check(report: dict) -> dict:
    """Heuristic: does the agent's summary agree with the outcome the system computed?

    Checks that it states no verdict the decision does not support, that a needs-review outcome
    is described as inconclusive or for review, and that any "<skill> level N" it states is the
    reported level. Passing is not proof of a faithful summary; failing is worth reading."""
    text = (report.get("summary") or "").lower()
    decision = report["decision"]
    problems = [f"states a verdict the outcome ({decision}) does not support: {p!r}"
                for p in CLAIMS.get(decision, []) if re.search(p, text)]
    if decision == "needs_review" and not re.search(r"not conclusive|inconclusive|review", text):
        problems.append("needs_review outcome not described as inconclusive or for human review")
    for skill, name in SKILL_NAMES.items():
        # only "<skill> level N" / "<skill> at level N", so job minimums in the reason are not read as levels
        for match in re.finditer(rf"\b{re.escape(name.lower())} (?:at |is at )?level (\d)", text):
            if int(match.group(1)) != report["levels"][skill]["level"]:
                problems.append(f"says {name} level {match.group(1)}, reported {report['levels'][skill]['level']}")
                break
    return {"consistent": not problems, "problems": problems, "heuristic": True}


def compare_drivers(agent_result: dict, code_result: dict) -> dict:
    """Same interview? And what the agent added in calls and time."""
    def ids(r):
        return [t["question_id"] for t in r["turns"]]
    agent = agent_result.get("agent") or {}
    return {
        "same_questions": ids(agent_result) == ids(code_result),
        "same_decision": agent_result["report"]["decision"] == code_result["report"]["decision"],
        "same_levels": all(agent_result["report"]["levels"][s]["level"] == code_result["report"]["levels"][s]["level"]
                           for s in SKILL_NAMES),
        "exchanges": {"agent": agent_result["exchanges"], "code": code_result["exchanges"]},
        "agent_steps": agent.get("steps"),
        "agent_llm_calls": (agent.get("interviewer_usage") or {}).get("llm_calls"),
        "llm_calls_total": {"agent": (agent_result.get("usage") or {}).get("llm_calls"),
                            "code": (code_result.get("usage") or {}).get("llm_calls")},
        "agent_llm_seconds": agent.get("llm_seconds"),
        "agent_tool_seconds": agent.get("tool_seconds"),
    }
