"""The declared experiment matrix (docs/experiment_matrix.json) must match the code.

If a preset changes, these tests fail, so the protocol cannot silently describe a system
that no longer exists. They check the declared overrides against the real presets, the
ADAPT12 control against the report's offline control, and the candidate allocation's balance.
"""

import json
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

from part4.candidates import load_candidates, truth_for
from shared.interfaces import SYSTEMS, InterviewConfig

ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / "docs" / "experiment_matrix.json").read_text(encoding="utf-8"))
BASELINE = json.loads((ROOT / "docs" / "evaluation_baseline.json").read_text(encoding="utf-8"))
IGNORED = {"system", "models"}   # names differ by definition; models are compared separately (A3)


def differences(config: InterviewConfig, reference: InterviewConfig) -> dict:
    return {f.name: getattr(config, f.name) for f in fields(InterviewConfig)
            if f.name not in IGNORED and getattr(config, f.name) != getattr(reference, f.name)}


def declared(name: str) -> dict:
    return MATRIX["conditions"][name]


@pytest.mark.parametrize("name", [n for n, c in MATRIX["conditions"].items()
                                  if c["preset"] and c["overrides"] is not None])
def test_declared_overrides_match_the_preset(name):
    condition = declared(name)
    actual = differences(SYSTEMS[condition["preset"]], SYSTEMS[MATRIX["reference_preset"]])
    assert actual == condition["overrides"]


def test_a3_differs_only_in_the_grader_model():
    a2, a3 = SYSTEMS["A2"], SYSTEMS["A3"]
    assert differences(a3, a2) == {}
    assert a3.models != a2.models and a3.models.grader != a2.models.grader


def test_adapt12_matches_the_report_offline_control():
    # The report's original offline control was SYSTEMS["A1"].with_(diagnosis="off", generic_follow_up=False);
    # report/make_figures.py now uses the ADAPT12 preset, which must be exactly that configuration.
    control = SYSTEMS["A1"].with_(diagnosis="off", generic_follow_up=False)
    assert differences(control, SYSTEMS["ADAPT12"]) == {}
    assert differences(control, SYSTEMS["A2"]) == declared("ADAPT12")["overrides"]
    assert control.fixed_count == SYSTEMS["B1"].fixed_count == 12


def test_revised_job_blind_condition_changes_only_retrieval():
    a2rev, a5rev = SYSTEMS[declared("A2-rev")["preset"]], SYSTEMS[declared("A5-rev")["preset"]]
    assert differences(a5rev, a2rev) == {"retrieval": "job-blind"}
    assert a2rev.to_dict()["diagnosis_prompt_version"] == a5rev.to_dict()["diagnosis_prompt_version"]


def test_a4_cannot_tell_reply_from_question_counting():
    """With one required confirmation both units agree, so A4 needs no question-unit twin."""
    assert SYSTEMS["A4"].misconception_confirmations == 1


def test_comparisons_name_declared_conditions():
    names = set(MATRIX["conditions"])
    for comparison in MATRIX["comparisons"]:
        if comparison.get("paired", True):
            assert set(comparison["conditions"]) <= names, comparison["id"]
    assert [c["id"] for c in MATRIX["comparisons"] if c["primary"]] == ["C1", "C2", "C3"]


def test_runs_name_declared_conditions():
    names = set(MATRIX["conditions"])
    for run, spec in MATRIX["runs"].items():
        if run in ("midsub", "rv1-sqlverify"):
            continue
        assert set(spec["conditions"]) | set(spec.get("optional", [])) <= names, run


def test_allocation_is_a_balanced_partition_of_the_frozen_candidates():
    shares = {k: MATRIX["allocation"][k] for k in ("nilkanta", "souradeep")}
    candidates = {c["id"]: c for c in load_candidates(ROOT / "data" / "candidates")}
    hashed = {Path(p).stem for p in BASELINE["candidate_files_sha256"]}
    everyone = shares["nilkanta"] + shares["souradeep"]
    assert len(everyone) == len(set(everyone)) == 60
    assert set(everyone) == set(candidates) == hashed

    jobs = {j["id"]: j for j in json.loads((ROOT / "data" / "jobs.json").read_text(encoding="utf-8"))}
    for ids in shares.values():
        chosen = [candidates[i] for i in ids]
        truths = [truth_for(c, jobs[c["job_id"]]) for c in chosen]
        assert set(Counter(c["job_id"] for c in chosen).values()) == {10}
        assert Counter(c["profile_type"] for c in chosen) == {"even": 15, "uneven": 15}
        assert Counter(t["fit"] for t in truths) == {True: 15, False: 15}
        assert sum(not c["misconceptions"] for c in chosen) == 3
        assert sum(c["safety_attack"] is not None for c in chosen) == 2
        for block in range(5):            # every block of 12 splits 6/6 and stays balanced
            in_block = [c for c in chosen if (int(c["id"][5:]) - 1) // 12 == block]
            assert len(in_block) == 6
            assert Counter(c["profile_type"] for c in in_block) == {"even": 3, "uneven": 3}


def test_allocation_follows_the_declared_rule():
    ids = sorted(MATRIX["allocation"]["nilkanta"] + MATRIX["allocation"]["souradeep"])
    for i, cid in enumerate(ids):
        k = (i % 12) // 3
        owner = "nilkanta" if k in (0, 3) else "souradeep"
        assert cid in MATRIX["allocation"][owner]


def test_annotation_strata_add_up():
    answers = MATRIX["annotation"]["answers"]
    assert sum(answers["strata"].values()) == answers["items"] == 30
    difficulty = MATRIX["annotation"]["difficulty"]
    assert difficulty["per_skill"] * 3 == difficulty["items"] == 21
    assert MATRIX["annotation"]["kappa_threshold"] == 0.6
