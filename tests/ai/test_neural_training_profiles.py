from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from shards_ai.ai import (
    NeuralModelConfig,
    NeuralTrainingProfile,
    load_active_neural_profile,
    load_active_training_profile,
    load_training_profile,
    next_training_profile_id,
    save_training_profile,
)


def test_load_training_profile_and_fingerprint(tmp_path):
    path = tmp_path / "v001.yaml"
    path.write_text(
        "profile_id: v001\n"
        "method: imitation\n"
        "dataset: data.jsonl\n"
        "output: model.pt\n"
        "seed: 7\n"
        "model:\n"
        "  state_hidden_dim: 64\n"
        "metadata:\n"
        "  purpose: test\n",
        encoding="utf-8",
    )

    profile = load_training_profile(path)

    assert profile.profile_id == "v001"
    assert profile.resolved_model_config() == replace(NeuralModelConfig(), state_hidden_dim=64)
    assert len(profile.fingerprint) == 64
    assert profile.resolve_path(profile.dataset).name == "data.jsonl"


def test_v005_control_profile_loads_with_neutral_scales():
    profile_dir = Path("configs/neural_training_profiles/candidates")
    control = load_training_profile(profile_dir / "exp00101-v005-fusion-control.yaml")

    assert control.metadata["architecture"] == "structured_semantic_v5_fusion_experiment"
    assert control.resolved_model_config().card_fusion_id_scale == 1.0
    assert control.resolved_model_config().card_fusion_semantic_scale == 1.0


def test_imitation_slice_matches_action_and_legal_action_bounds():
    from shards_ai.ai.neural_training import matches_imitation_slice

    record = {
        "chosen_action": {"action_type": "play_card"},
        "action_representations": [{}, {}, {}, {}, {}],
    }
    assert matches_imitation_slice(
        record, action_types=frozenset({"play_card"}), min_legal_actions=5, max_legal_actions=8
    )
    assert not matches_imitation_slice(
        record, action_types=frozenset({"play_card"}), min_legal_actions=6, max_legal_actions=8
    )
    assert not matches_imitation_slice(
        record, action_types=frozenset({"activate_champion"}), min_legal_actions=5, max_legal_actions=8
    )

    record["teacher_action_type"] = "play_card"
    record["chosen_action"] = {"action_type": "activate_champion"}
    assert matches_imitation_slice(
        record, action_types=frozenset({"play_card"}), min_legal_actions=5, max_legal_actions=8
    )


def test_save_training_profile_round_trips(tmp_path):
    profile = NeuralTrainingProfile(profile_id="v002", dataset="data.jsonl", output="model.pt", parent_profile_id="v001")
    loaded = load_training_profile(save_training_profile(profile, tmp_path / "profile.yaml"))
    assert loaded == profile


def test_active_profile_and_next_version(tmp_path):
    directory = tmp_path / "profiles"
    directory.mkdir()
    save_training_profile(NeuralTrainingProfile(profile_id="v001", dataset="data", output="one.pt"), directory / "v001.yaml")
    save_training_profile(NeuralTrainingProfile(profile_id="v007", dataset="data", output="seven.pt"), directory / "v007.yaml")
    active = directory / "active.yaml"
    active.write_text("schema_version: 1\nactive_profile_id: v007\n", encoding="utf-8")

    assert load_active_training_profile(active).profile_id == "v007"
    assert next_training_profile_id(directory) == "v008"


def test_active_neural_profile_resolves_versioned_checkpoint(tmp_path):
    directory = tmp_path / "neural_profiles"
    directory.mkdir()
    checkpoint = directory / "v001.pt"
    checkpoint.write_bytes(b"checkpoint")
    active = directory / "active.yaml"
    active.write_text("schema_version: 1\nactive_profile_id: v001\n", encoding="utf-8")

    profile = load_active_neural_profile(active)

    assert profile.profile_id == "v001"
    assert profile.checkpoint_path == checkpoint


def test_batched_validation_ranges_cover_games_without_overlap():
    from scripts.validate_neural_profile_batched import batch_ranges

    assert batch_ranges(45, 20) == [(0, 20), (20, 40), (40, 45)]


def test_quality_panel_includes_fixed_neural_references_without_random():
    from argparse import Namespace
    from pathlib import Path

    from scripts.validate_neural_profile import _panel

    opponents, _heuristics, neural_profiles, hybrid_profiles = _panel(
        Namespace(
            profile_dir=Path("configs/neural_training_profiles"),
            profile_v008=Path("configs/heuristic_profiles/v008.yaml"),
            profile_hybrid_v001=Path("configs/hybrid_profiles/hybrid-v001.yaml"),
            profile_hybrid_v003=Path("configs/hybrid_profiles/hybrid-v003.yaml"),
            profile_hybrid_v004=Path("configs/hybrid_profiles/hybrid-v004.yaml"),
            profile_hybrid_v005=Path("configs/hybrid_profiles/hybrid-v005.yaml"),
            profile_hybrid_v006=Path("configs/hybrid_profiles/hybrid-v006.yaml"),
        ),
        "exp-candidate",
    )

    assert opponents == ["hybrid:v006", "hybrid:v004", "hybrid:v005", "v008", "hybrid:v001", "hybrid:v003"]
    assert set(neural_profiles) == set()
    assert set(hybrid_profiles) == {"v001", "v003", "v004", "v005", "v006"}


def test_validation_rule_accepts_positive_mean_with_secondary_regressions():
    from scripts.validate_neural_profile import acceptance_decision

    assert acceptance_decision({
        "hybrid:v006": {"delta_win_rate": 0.01},
        "hybrid:v004": {"delta_win_rate": 0.01},
        "hybrid:v005": {"delta_win_rate": 0.01},
        "v008": {"delta_win_rate": 0.06},
        "hybrid:v001": {"delta_win_rate": 0.01},
        "hybrid:v003": {"delta_win_rate": 0.01},
    })


def test_validation_rule_requires_complete_panel_and_positive_mean_progress():
    from scripts.validate_neural_profile import acceptance_decision

    assert not acceptance_decision({"v007": {"delta_win_rate": 0.0}})
    assert acceptance_decision({
        "hybrid:v006": {"delta_win_rate": 0.0},
        "hybrid:v004": {"delta_win_rate": 0.0},
        "hybrid:v005": {"delta_win_rate": 0.0},
        "v008": {"delta_win_rate": 0.0},
        "hybrid:v001": {"delta_win_rate": 0.0},
        "hybrid:v003": {"delta_win_rate": 0.01},
    })


def test_validation_rule_ignores_random_when_weighted_mean_is_positive():
    from scripts.validate_neural_profile import acceptance_decision

    assert acceptance_decision({
        "hybrid:v006": {"delta_win_rate": 0.01},
        "hybrid:v004": {"delta_win_rate": 0.01},
        "hybrid:v005": {"delta_win_rate": 0.01},
        "v008": {"delta_win_rate": 0.03},
        "hybrid:v001": {"delta_win_rate": 0.01},
        "hybrid:v003": {"delta_win_rate": 0.01},
    })
    assert acceptance_decision({
        "hybrid:v006": {"delta_win_rate": 0.01},
        "hybrid:v004": {"delta_win_rate": 0.01},
        "hybrid:v005": {"delta_win_rate": 0.01},
        "v008": {"delta_win_rate": -0.01},
        "hybrid:v001": {"delta_win_rate": 0.01},
        "hybrid:v003": {"delta_win_rate": 0.05},
    })


def test_validation_rule_allows_v008_regression_when_weighted_mean_is_positive():
    from scripts.validate_neural_profile import acceptance_metrics

    metrics = acceptance_metrics({
        "hybrid:v006": {"delta_win_rate": 0.01},
        "hybrid:v004": {"delta_win_rate": 0.01},
        "hybrid:v005": {"delta_win_rate": 0.01},
        "v008": {"delta_win_rate": -0.04},
        "hybrid:v001": {"delta_win_rate": 0.01},
        "hybrid:v003": {"delta_win_rate": 0.05},
    })

    assert metrics["accepted"]
    assert metrics["mean_delta_win_rate"] > 0.0


def test_validation_rule_uses_configured_opponent_weights_and_active_references():
    from scripts.validate_neural_profile import acceptance_metrics

    metrics = acceptance_metrics(
        {
            "hybrid:v006": {"delta_win_rate": 0.01},
            "hybrid:v004": {"delta_win_rate": 0.02},
            "hybrid:v005": {"delta_win_rate": 0.03},
            "v008": {"delta_win_rate": 0.07},
            "hybrid:v001": {"delta_win_rate": 0.04},
            "hybrid:v003": {"delta_win_rate": 0.05},
        },
        {"buy": {"delta": 0.04, "weight": 1.0}, "play": {"delta": 0.03, "weight": 1.0}},
    )

    assert metrics["accepted"]
    assert metrics["mean_delta_win_rate"] == pytest.approx(0.0359090909)
    assert metrics["opponent_weights"] == {
        "hybrid:v006": 1.0,
        "hybrid:v004": 1.0,
        "hybrid:v005": 1.0,
        "v008": 1.0,
        "hybrid:v001": 0.75,
        "hybrid:v003": 0.75,
    }
    assert "random" not in metrics["opponent_weights"]


def test_validation_rule_rejects_an_incomplete_neural_reference_panel():
    from scripts.validate_neural_profile import acceptance_metrics

    metrics = acceptance_metrics({
        "v008": {"delta_win_rate": 0.01},
        "neural:v006": {"delta_win_rate": 0.10},
    })

    assert not metrics["accepted"]
    assert metrics["reason"] == "missing_neural_references"


def test_validation_output_shows_precise_candidate_and_reference_rates():
    from scripts.validate_neural_profile import format_validation_line

    line = format_validation_line(
        "v008",
        {
            "candidate": {"wins": 27, "games": 100, "win_rate": 0.27},
            "reference": {"wins": 26, "games": 100, "win_rate": 0.26},
            "delta_win_rate": 0.01,
            "improved": True,
            "not_regressed": True,
        },
        "v002",
        "v001",
    )

    assert line == "v008 : v002 27.00% (27/100) | v001 26.00% (26/100) | delta +1.00% | PROGRES"


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 0), ("torch_threads", 0), ("learning_rate", 0)],
)
def test_invalid_training_profile_is_rejected(tmp_path, field, value):
    values = {"profile_id": "bad", "dataset": "data.jsonl", "output": "model.pt", field: value}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_profile(path)


def test_ppo_training_profile_accepts_game_budget_and_opponents(tmp_path):
    path = tmp_path / "v002.yaml"
    path.write_text(
        "profile_id: v002\n"
        "parent_profile_id: v001\n"
        "method: ppo\n"
        "output: artifacts/neural_training/checkpoint.pt\n"
        "initial_checkpoint: configs/neural_profiles/v001.pt\n"
        "total_games: 12\n"
        "games_per_update: 4\n"
        "optimization_epochs: 2\n"
        "minibatch_size: 8\n"
        "opponents:\n"
        "  random: 0.333333\n"
        "  v007: 0.333333\n"
        "  v008: 0.333334\n",
        encoding="utf-8",
    )

    profile = load_training_profile(path)

    assert profile.method == "ppo"
    assert profile.dataset is None
    assert profile.total_games == 12
    assert profile.evaluation_games == 64
    assert set(profile.opponents) == {"random", "v007", "v008"}
