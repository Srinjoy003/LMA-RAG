"""Pilot inventory must not treat a partial/duplicate run as completed."""

import json
import sys

import pytest

from test_step5_evaluation import load_script


def test_complete_inventory_requires_every_planned_pair_exactly_once():
    module = load_script("audit_pilot")
    plan = {"systems": ["A2", "A2E"], "candidate_ids": ["VAL-P-001", "VAL-P-002"]}
    results = {s: [{"candidate_id": cid} for cid in plan["candidate_ids"]] for s in plan["systems"]}
    counts = module.completeness(results, plan)
    assert counts == {"expected": 4, "saved": 4, "missing": [], "unexpected": [], "duplicates": []}
    results["A2E"][1] = {"candidate_id": "VAL-P-001"}
    counts = module.completeness(results, plan)
    assert counts["missing"] == [("A2E", "VAL-P-002")]
    assert counts["duplicates"] == [("A2E", "VAL-P-001")]


def test_final_evaluation_candidates_cannot_complete_the_pilot():
    module = load_script("audit_pilot")
    counts = module.completeness({"A2": [{"candidate_id": "CAND-001"}]},
                                 {"systems": ["A2"], "candidate_ids": ["VAL-P-001"]})
    assert counts["missing"] and counts["unexpected"]


def test_runtime_includes_time_across_midnight():
    assert load_script("audit_pilot").elapsed_seconds({
        "started_at": "2026-11-14T23:59:00Z", "finished_at": "2026-11-15T00:01:00Z"}) == 120


def test_missing_live_results_are_not_replaced_by_a_stub_report(tmp_path, monkeypatch):
    module = load_script("audit_pilot")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/step8_pilot_plan.json").write_text(json.dumps({}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    out = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit_pilot", "--json", str(out)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2 and not out.exists()
