import json

import pytest

from shards_ai.experimentation import (
    ExperimentManifest,
    ExperimentStatus,
    render_experiment_report,
    validate_changed_paths,
)
from shards_ai.experimentation.policy import evaluate_performance_gate, validate_campaign_settings
from shards_ai.experimentation.diversity import EXPERIMENT_FAMILIES, family_guidance


def test_manifest_round_trips_to_json(tmp_path):
    manifest = ExperimentManifest(
        experiment_id="exp-00001",
        campaign_id="campaign-test",
        experiment_kind="quality",
        parent_commit="abc123",
        parent_profile="v008",
        hypothesis="augmenter le poids des achats",
        status=ExperimentStatus.REJECTED,
    )
    path = tmp_path / "manifest.json"
    manifest.write_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "rejected"
    assert data["hypothesis"] == "augmenter le poids des achats"
    assert data["training_budget_seconds"] == 2400


def test_report_is_human_readable_for_rejected_experiment():
    manifest = ExperimentManifest(
        experiment_id="exp-00002",
        campaign_id="campaign-test",
        experiment_kind="quality",
        parent_commit="abc123",
        parent_profile="v008",
        hypothesis="changer l'embedding",
        status=ExperimentStatus.REJECTED,
        validation={"v008": {"delta_win_rate": -0.02}},
    )

    report = render_experiment_report(manifest, {"analysis": "régression contre V008"})

    assert "# Expérience exp-00002" in report
    assert "régression contre V008" in report
    assert "delta_win_rate" in report


def test_protected_game_and_heuristic_paths_are_rejected():
    with pytest.raises(ValueError, match="protected paths"):
        validate_changed_paths(["shards_ai/game/game.py"])
    with pytest.raises(ValueError, match="protected paths"):
        validate_changed_paths(["configs/heuristic_profiles/v008.yaml"])
    with pytest.raises(ValueError, match="protected paths"):
        validate_changed_paths(["configs/neural_profiles/active.yaml"])
    with pytest.raises(ValueError, match="protected paths"):
        validate_changed_paths(["configs/neural_training_profiles/v009.yaml"])


def test_neural_paths_are_allowed():
    validate_changed_paths([
        "shards_ai/ai/neural_model.py",
        "configs/neural_training_profiles/candidates/exp.yaml",
        "doc/Experiments/exp-00001.md",
    ])


def test_campaign_invariants_are_protected():
    validate_campaign_settings({
        "seed": 104,
        "opponents": ["random", "v007", "v008"],
        "baseline_profile": "v008",
        "acceptance_rule": "strict",
    })

    with pytest.raises(ValueError, match="baseline"):
        validate_campaign_settings({
            "seed": 104,
            "opponents": ["v008"],
            "baseline_profile": "v007",
            "acceptance_rule": "strict",
        })


def test_manifest_default_budget_is_one_hour_with_training_split():
    manifest = ExperimentManifest(
        experiment_id="exp-00003",
        campaign_id="campaign-test",
        experiment_kind="quality",
        parent_commit="abc123",
        parent_profile="v008",
        hypothesis="calibration",
    )

    assert manifest.budget_seconds == 3600
    assert manifest.training_budget_seconds == 2400
    assert manifest.screening_budget_seconds == 750
    assert manifest.overhead_budget_seconds == 450


def test_performance_gate_rejects_a_large_elapsed_time_regression():
    gate = evaluate_performance_gate({
        "baseline": {"elapsed_seconds": 100.0},
        "candidate": {"elapsed_seconds": 106.0},
    })

    assert gate["available"]
    assert not gate["accepted"]
    assert gate["max_regression"] == pytest.approx(0.06)


def test_performance_gate_accepts_a_small_throughput_regression():
    gate = evaluate_performance_gate({
        "baseline": {"throughput": 100.0},
        "candidate": {"throughput": 98.0},
    })

    assert gate["accepted"]


def test_diversity_guidance_prefers_non_ppo_after_two_ppo_runs():
    assert {"data", "objective", "inference"}.issubset(EXPERIMENT_FAMILIES)
    guidance = family_guidance([
        {"experiment_family": "ppo"},
        {"experiment_family": "ppo"},
        {"experiment_family": "imitation"},
    ])

    assert guidance["last_family"] == "imitation"
    assert guidance["consecutive_last_family"] == 1
    assert guidance["recommendation"] == "choose freely"

    guidance = family_guidance([
        {"experiment_family": "ppo"},
        {"experiment_family": "ppo"},
    ])
    assert "non-PPO" in guidance["recommendation"]
