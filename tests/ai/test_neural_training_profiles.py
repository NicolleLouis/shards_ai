from dataclasses import replace

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


def test_validation_rule_accepts_positive_mean_with_secondary_regressions():
    from scripts.validate_neural_profile import acceptance_decision

    assert acceptance_decision({
        "random": {"delta_win_rate": -0.02},
        "v007": {"delta_win_rate": -0.01},
        "v008": {"delta_win_rate": 0.06},
    })


def test_validation_rule_requires_v008_and_positive_mean_progress():
    from scripts.validate_neural_profile import acceptance_decision

    assert not acceptance_decision({"random": {"delta_win_rate": 0.0}})
    assert acceptance_decision({
        "random": {"delta_win_rate": 0.01},
        "v007": {"delta_win_rate": 0.0},
        "v008": {"delta_win_rate": 0.0},
    })


def test_validation_rule_rejects_large_secondary_regression():
    from scripts.validate_neural_profile import acceptance_decision

    assert not acceptance_decision({
        "random": {"delta_win_rate": -0.06},
        "v007": {"delta_win_rate": 0.03},
        "v008": {"delta_win_rate": 0.03},
    })
    assert not acceptance_decision({
        "random": {"delta_win_rate": 0.10},
        "v007": {"delta_win_rate": 0.10},
        "v008": {"delta_win_rate": -0.01},
    })


def test_validation_rule_uses_opponent_mean_and_ignores_category_weights():
    from scripts.validate_neural_profile import acceptance_metrics

    metrics = acceptance_metrics(
        {
            "random": {"delta_win_rate": -0.02},
            "v007": {"delta_win_rate": -0.02},
            "v008": {"delta_win_rate": 0.07},
        },
        {"buy": {"delta": 0.04, "weight": 1.0}, "play": {"delta": 0.03, "weight": 1.0}},
    )

    assert metrics["accepted"]
    assert metrics["mean_delta_win_rate"] == pytest.approx(0.01)


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
