"""Step 5: versioned runs, explicit candidates, new presets and the extended evaluation.

All offline: stub grader and simulator, temporary run folders, synthetic edits.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from engine_support import ROOT, real_bank
from part4 import runner
from part4.candidates import candidate_digest, load_candidates, make_candidates, save_candidates
from part4.manifest import (
    FAILURES, MANIFEST, RunManifest, RunManifestError, bank_hashes, failure_counts, file_sha256, text_sha256,
)
from part4.metrics import (
    NOT_MEASURED, check_run, compare, evidence_metrics, finding_status, load_results, misconception_metrics,
    premature_rates, summarise,
)
from part4.runner import Setup, run_experiment
from shared.interfaces import SYSTEMS

JOBS = {j["id"]: j for j in real_bank().jobs}
QUIET = {"report": lambda *_: None}


@pytest.fixture(scope="module")
def candidates():
    return make_candidates(real_bank(), n=12, seed=11)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / ("report" if name == "make_figures" else "scripts")
                                                  / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Manifests and refusals
# --------------------------------------------------------------------------

def test_first_run_writes_a_manifest_and_tags_results(tmp_path, candidates):
    run_experiment(["B1", "A2E"], candidates[:2], JOBS, Setup(), tmp_path / "rv-test", operator="tester", **QUIET)
    manifest = json.loads((tmp_path / "rv-test" / MANIFEST).read_text())
    assert set(manifest["conditions"]) == {"B1", "A2E"}
    assert manifest["conditions"]["A2E"]["diagnosis_prompt_version"] == "evidence-v1"
    assert set(manifest["candidates"]) == {c["id"] for c in candidates[:2]}
    assert manifest["fingerprint"]["bank_files_sha256"] == bank_hashes(ROOT)
    assert "src/shared/evidence.py" in manifest["fingerprint"]["prompt_sources_sha256"]
    assert manifest["sessions"][0]["operator"] == "tester"
    result = load_results(tmp_path / "rv-test")["A2E"][0]
    assert result["run"] == {"name": "rv-test", "fingerprint": manifest["fingerprint_id"]}
    assert check_run(tmp_path / "rv-test", load_results(tmp_path / "rv-test")) == []


def test_resuming_and_adding_conditions_or_candidates_is_allowed(tmp_path, candidates):
    run_dir = tmp_path / "rv"
    run_experiment(["B1"], candidates[:2], JOBS, Setup(), run_dir, **QUIET)
    counts = run_experiment(["B1", "A4"], candidates[:3], JOBS, Setup(), run_dir, **QUIET)
    assert counts == {"done": 4, "skipped": 2, "failed": 0, "excluded": 0}
    assert len(json.loads((run_dir / MANIFEST).read_text())["sessions"]) == 2


def test_an_old_folder_without_a_manifest_is_never_extended(tmp_path, candidates):
    legacy = tmp_path / "midsub" / "A2" / "data-analyst"
    legacy.mkdir(parents=True)
    (legacy / "CAND-001.json").write_text("{}")
    with pytest.raises(RunManifestError, match="no manifest.json"):
        run_experiment(["A2"], candidates[:1], JOBS, Setup(), tmp_path / "midsub", **QUIET)
    assert sorted(p.name for p in (tmp_path / "midsub").rglob("*.json")) == ["CAND-001.json"]


def test_a_changed_condition_configuration_is_refused(tmp_path, candidates, monkeypatch):
    run_dir = tmp_path / "rv"
    run_experiment(["A2"], candidates[:1], JOBS, Setup(), run_dir, **QUIET)
    monkeypatch.setitem(SYSTEMS, "A2", SYSTEMS["A2"].with_(uncertainty_threshold=0.5))
    with pytest.raises(RunManifestError, match="uncertainty_threshold"):
        run_experiment(["A2"], candidates[1:2], JOBS, Setup(), run_dir, **QUIET)
    assert len(load_results(run_dir)["A2"]) == 1


def test_a_changed_candidate_key_is_refused(tmp_path, candidates):
    run_dir = tmp_path / "rv"
    run_experiment(["B1"], candidates[:1], JOBS, Setup(), run_dir, **QUIET)
    edited = copy.deepcopy(candidates[0])
    edited["misconceptions"] = []
    with pytest.raises(RunManifestError, match="candidate file differs"):
        run_experiment(["B1"], [edited], JOBS, Setup(), run_dir, rerun=True, **QUIET)


def make_tree(root: Path, code: str = "x = 1\n") -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src/module.py").write_bytes(code.encode())    # exact bytes: write_text translates "\n" on Windows
    for name in ("data/questions/py.json", "data/jobs.json"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("[]")
    return root


def test_changed_code_bank_or_setup_is_refused(tmp_path):
    tree = make_tree(tmp_path / "repo")
    setup = {"grader": "stub", "simulator": "stub", "simulator_model": "stub"}
    RunManifest.open(tmp_path / "run", code_root=tree, bank_root=tree, setup=setup).save()
    RunManifest.open(tmp_path / "run", code_root=tree, bank_root=tree, setup=setup)       # unchanged: fine
    with pytest.raises(RunManifestError, match="setup.grader"):
        RunManifest.open(tmp_path / "run", code_root=tree, bank_root=tree, setup={**setup, "grader": "gemini"})
    (tree / "data/jobs.json").write_text('[{"id": "x"}]')
    with pytest.raises(RunManifestError, match="bank_files_sha256.data/jobs.json"):
        RunManifest.open(tmp_path / "run", code_root=tree, bank_root=tree, setup=setup)
    (tree / "data/jobs.json").write_text("[]")
    (tree / "src/module.py").write_bytes(b"x = 2\n")
    with pytest.raises(RunManifestError, match="code"):
        RunManifest.open(tmp_path / "run", code_root=tree, bank_root=tree, setup=setup)


def test_line_endings_do_not_change_the_fingerprint(tmp_path):
    assert text_sha256(b"a\r\nb\r\n") == text_sha256(b"a\nb\n")
    lf, crlf = make_tree(tmp_path / "lf"), make_tree(tmp_path / "crlf", code="x = 1\r\n")
    setup = {"grader": "stub"}
    a = RunManifest.open(tmp_path / "r1", code_root=lf, bank_root=lf, setup=setup)
    b = RunManifest.open(tmp_path / "r2", code_root=crlf, bank_root=crlf, setup=setup)
    assert a.fingerprint_id == b.fingerprint_id


def test_frozen_evaluation_candidates_still_match_the_baseline():
    """Same hashes on Windows (CRLF checkout) and macOS: the 60 keys were not regenerated."""
    baseline = json.loads((ROOT / "docs/evaluation_baseline.json").read_text(encoding="utf-8"))
    for name, digest in baseline["candidate_files_sha256"].items():
        assert file_sha256(ROOT / name) == digest, name


def test_failures_are_logged_and_excluded_after_three_attempts(tmp_path, candidates, monkeypatch):
    calls = []

    def failing(system, candidate, *args, **kwargs):
        calls.append(candidate["id"])
        return {"stop_reason": "error", "errors": ["RuntimeError: boom"]}
    monkeypatch.setattr(runner, "run_one", failing)
    run_dir = tmp_path / "rv"
    for _ in range(4):
        counts = run_experiment(["A2"], candidates[:1], JOBS, Setup(), run_dir, **QUIET)
    assert calls == [candidates[0]["id"]] * 3
    assert counts == {"done": 0, "skipped": 0, "failed": 0, "excluded": 1}
    assert failure_counts(run_dir) == {("A2", candidates[0]["id"]): 3}


def test_a_quota_stop_is_a_resume_not_a_failed_attempt(tmp_path, candidates, monkeypatch):
    monkeypatch.setattr(runner, "run_one", lambda *a, **k: {"stop_reason": "error",
                                                            "errors": ["LLMQuotaError: Quota per day"]})
    run_experiment(["A2"], candidates[:2], JOBS, Setup(), tmp_path / "rv", **QUIET)
    assert not (tmp_path / "rv" / FAILURES).exists()


# --------------------------------------------------------------------------
# Candidates and presets
# --------------------------------------------------------------------------

def test_validation_populations_have_their_own_ids(tmp_path):
    pilot = make_candidates(real_bank(), n=12, seed=3102, prefix="VAL-P")
    assert [c["id"] for c in pilot][:2] == ["VAL-P-001", "VAL-P-002"]
    with pytest.raises(ValueError, match="prefix"):
        make_candidates(real_bank(), n=1, prefix="EVAL")
    save_candidates(pilot, tmp_path)
    assert load_candidates(tmp_path, ["VAL-P-003", "VAL-P-001"]) == [pilot[2], pilot[0]]
    with pytest.raises(KeyError, match="VAL-P-099"):
        load_candidates(tmp_path, ["VAL-P-099"])


def test_the_committed_pilot_population_is_the_declared_one():
    committed = load_candidates(ROOT / "data/candidates_validation/pilot")
    regenerated = make_candidates(real_bank(), n=12, seed=3102, prefix="VAL-P")
    assert [candidate_digest(c) for c in committed] == [candidate_digest(c) for c in regenerated]


def test_candidate_digest_ignores_formatting():
    c = {"id": "CAND-001", "b": [1, 2], "a": "x"}
    assert candidate_digest(c) == candidate_digest(json.loads(json.dumps(c, indent=4)))


def test_allocation_and_explicit_ids_are_resolved(tmp_path):
    module = load_script("run_experiment")
    parser = argparse.ArgumentParser()
    ns = argparse.Namespace(ids=None, ids_file=None, allocation="nilkanta")
    ids = module.chosen_ids(ns, parser)
    assert len(ids) == 30 and ids[0] == "CAND-001"
    listing = tmp_path / "ids.txt"
    listing.write_text("# pilot\nVAL-P-001\n\nVAL-P-002\n")
    assert module.chosen_ids(argparse.Namespace(ids=None, ids_file=listing, allocation=None), parser) == \
        ["VAL-P-001", "VAL-P-002"]
    with pytest.raises(SystemExit):
        module.chosen_ids(argparse.Namespace(ids=["CAND-001"], ids_file=None, allocation="nilkanta"), parser)


def test_new_presets_run_end_to_end(tmp_path, candidates):
    run_experiment(["ADAPT12", "A5E"], candidates[:2], JOBS, Setup(), tmp_path / "rv", **QUIET)
    results = load_results(tmp_path / "rv")
    assert all(r["exchanges"] == 12 and len(r["turns"]) == 12 for r in results["ADAPT12"])
    assert all(r["config"]["retrieval"] == "job-blind" and r["config"]["diagnostic_evidence"]
               for r in results["A5E"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stub_results(tmp_path_factory, candidates):
    run_dir = tmp_path_factory.mktemp("stub") / "rv"
    run_experiment(["B1", "A2E", "A4"], candidates, JOBS, Setup(), run_dir, **QUIET)
    return run_dir, load_results(run_dir)


def test_disputed_findings_are_never_counted_as_confirmed(stub_results):
    _, by_system = stub_results
    results = copy.deepcopy(by_system["A2E"][:1])
    r = results[0]
    planted = r["truth"]["misconceptions"][0]
    r["report"]["misconceptions"] = [
        {"id": planted, "confirmed": False, "status": "disputed", "evidence_turns": [1, 2], "vote_share": 1.0},
        {"id": "py.legacy.finding", "confirmed": True, "evidence_turns": [1], "vote_share": 1.0}]
    m = misconception_metrics(results)
    assert m["disputed"] == 1 and m["disputed_planted"] == 1 and m["confirmed"] == 1
    assert m["found"] == 0 and m["found_named"] == 1
    assert finding_status({"confirmed": True}) == "confirmed" and finding_status({"confirmed": False}) == "suspected"


def test_evidence_and_cost_metrics(stub_results):
    _, by_system = stub_results
    results = copy.deepcopy(by_system["A2E"][:2])
    results[0]["turns"][0]["diagnosis"] = {"cause": "gap", "evidence_rejected": "missing quote",
                                           "contradicting": [{"misconception_id": "x"}]}
    results[0]["usage"] = {"llm_calls": 10, "api_calls": 4}
    m = evidence_metrics(results)
    assert m["quote_rejections"] == 1 and m["contradiction_claims"] == 1
    s = summarise(results)
    assert s["mean_llm_calls"] == 5 and s["mean_api_calls"] == 2
    assert 0 <= s["topic_coverage"] <= 1


def test_premature_rates_carry_denominators(stub_results):
    _, by_system = stub_results
    rates = premature_rates(by_system["A2E"])
    confident = sum(r["stop_reason"] == "confident" for r in by_system["A2E"])
    assert sum(v["assessed"] for v in rates["by_skill"].values()) == 3 * confident
    assert sum(v["assessed"] for v in rates["by_true_level"].values()) == 3 * confident


def test_compare_pairs_only_its_two_systems(stub_results):
    _, by_system = stub_results
    uneven = {**by_system, "A4": by_system["A4"][:3]}          # an incomplete third system
    result = compare(uneven, "A2E", "B1", ["skill_accuracy", "precision", "human_supported_precision"],
                     samples=200)
    assert result["candidates"] == len(by_system["B1"])
    acc = result["endpoints"]["skill_accuracy"]
    assert acc["difference"] == pytest.approx(acc["A2E"] - acc["B1"])
    assert acc["interval"][0] <= acc["difference"] <= acc["interval"][1]
    assert result["endpoints"]["precision"]["difference"] is None          # B1 confirms nothing
    assert result["endpoints"]["precision"]["undefined_draws"] == 200
    assert result["endpoints"]["human_supported_precision"] == {"status": NOT_MEASURED["human_supported_precision"]}
    again = compare(uneven, "A2E", "B1", ["skill_accuracy"], samples=200)
    assert again["endpoints"]["skill_accuracy"] == acc


def test_check_run_flags_results_from_another_manifest(stub_results, tmp_path):
    run_dir, by_system = stub_results
    tampered = copy.deepcopy(by_system)
    tampered["B1"][0]["run"]["fingerprint"] = "other"
    assert any("not made under this run's manifest" in p for p in check_run(run_dir, tampered))
    assert any("no manifest" in p for p in check_run(tmp_path, by_system))


def test_evaluate_runs_the_declared_comparisons(stub_results, tmp_path, capsys, monkeypatch):
    run_dir, _ = stub_results
    module = load_script("evaluate")
    monkeypatch.setattr(module, "RESULTS_DIR", run_dir.parent)
    out = tmp_path / "metrics.json"
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--run", run_dir.name, "--method", "A2E", "--comparisons",
                                     "--samples", "100", "--json", str(out)])
    assert module.main() == 0
    text = capsys.readouterr().out
    assert "C1 (RQ1): A2E vs B1" in text and "C3 (RQ2): A2E vs A4" in text
    assert "human_supported_precision: not measured" in text
    saved = json.loads(out.read_text())
    assert {c["id"] for c in saved["comparisons"]} == {"C1", "C3"} and saved["provenance"] == []


# --------------------------------------------------------------------------
# Report figures: explicit bank
# --------------------------------------------------------------------------

def test_figures_refuse_an_unnamed_or_mismatched_bank(tmp_path, stub_results):
    pytest.importorskip("matplotlib")
    module = load_script("make_figures")
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):                     # no manifest and no label: refused
        module.check_bank(tmp_path / "midsub", ROOT, None, parser)
    assert module.check_bank(tmp_path / "midsub", ROOT, "historical", parser)[1] == "historical"
    run_dir, _ = stub_results
    assert module.check_bank(run_dir, ROOT, None, parser)[0] == ROOT.resolve()
    other = make_tree(tmp_path / "other")
    with pytest.raises(SystemExit):
        module.check_bank(run_dir, other, "reviewed", parser)
