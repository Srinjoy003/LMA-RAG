"""Pieces shared by the drivers: required topics, turn and result assembly, saving.

Used by the session (and so the code driver) and by the agent driver, so all of them
write the same result format (schemas/interview_result.schema.json).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPIC_ID = re.compile(r"^(py|sql|ds)-[a-z-]+$")


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def required_topics_for(job: dict, config, components, errors: list[str]) -> list[str]:
    """Topic extraction when it is on and an extractor is given; otherwise jobs.json."""
    if config.use_topic_extraction and components.topic_extractor is not None:
        try:
            topics = [t for t in components.topic_extractor(job.get("description", "")) if TOPIC_ID.match(t)]
            if topics:
                return list(dict.fromkeys(topics))
            errors.append("topic extractor returned no topics; used required_topics from jobs.json")
        except Exception as e:
            errors.append(f"topic extractor failed ({type(e).__name__}: {e}); used required_topics from jobs.json")
    return list(job.get("required_topics", []))


def build_turn(found: dict, answer: str, exchange: dict, update: dict) -> dict:
    """One entry of `turns`. `found` is the search_question_bank result, `exchange` the
    result of the follow-up flow, `update` the update_scorecard result."""
    return {
        "index": found["turn"], "skill": found["skill"], "level": found["level"],
        "question_id": found["question_id"], "question_source": found["source"],
        "topics": found["topics"], "selection_reason": found["selection_reason"],
        "prompt": found["prompt"], "answer": answer,
        "diagnosis": exchange["diagnosis"], "follow_ups": exchange["follow_ups"],
        "grade": update["grade"], "provisional_grade": exchange["provisional"],
        "evidence_weight": update["evidence_weight"], "scorecard_after": update["scorecard"],
    }


def attach_tool_calls(turns: list[dict], log: list[dict]) -> None:
    """Give each turn the tool calls made for it. Calls after the last question
    (end_interview) go to the last turn; refused calls belong to the turn they were for."""
    for turn in turns:
        turn["tool_calls"] = []
    for call in log:
        if not turns:
            break
        index = min(max(call["turn"], 1), len(turns)) - 1
        turns[index]["tool_calls"].append({k: call[k] for k in ("name", "args", "ok", "error")})


def build_result(*, job: dict, config, candidate_id: str | None, started: datetime, finished: datetime,
                 required: list[str], turns: list[dict], stop_reason: str, report: dict,
                 errors: list[str], tool_log: list[dict]) -> dict:
    """The interview result dict. Attaches the tool-call log to the turns."""
    attach_tool_calls(turns, tool_log)
    return {
        "interview_id": f"{config.system}-{job['id']}-{candidate_id or 'live'}-{config.seed}",
        "system": config.system,
        "config": json.loads(json.dumps(config.to_dict())),
        "job_id": job["id"],
        "candidate_id": candidate_id,
        "started_at": iso(started),
        "finished_at": iso(finished),
        "required_topics": required,
        "turns": turns,
        "exchanges": len(turns) + sum(len(t["follow_ups"]) for t in turns),
        "stop_reason": stop_reason,
        "report": report,
        "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "by_model": {}},
        "errors": errors,
    }


def save_result(result: dict, root: str | Path = ROOT, *, folder: str | Path | None = None) -> Path:
    """Write results/<system>/<job_id>/<candidate_id>.json (live interviews: live-<timestamp>).
    With `folder`, write <folder>/<name>.json instead (the app uses results/live)."""
    name = result.get("candidate_id") or "live-" + re.sub(r"\D", "", result["started_at"])
    config = result.get("config") or {}
    if config.get("driver", "code") != "code":      # keep agent and code runs of one candidate apart
        name += f"-{config['driver']}"
    if config.get("seed"):
        name += f"-seed{config['seed']}"
    if folder is not None:
        path = Path(folder) / f"{name}.json"
    else:
        path = Path(root) / "results" / result["system"] / result["job_id"] / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
