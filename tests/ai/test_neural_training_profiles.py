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


def test_validation_rule_requires_progress_without_regression():
    from scripts.validate_neural_profile import acceptance_decision

    assert acceptance_decision({"random": {"delta_win_rate": 0.01}, "v007": {"delta_win_rate": 0.0}})
    assert not acceptance_decision({"random": {"delta_win_rate": 0.0}})
    assert not acceptance_decision({"random": {"delta_win_rate": 0.01}, "v007": {"delta_win_rate": -0.01}})


@pytest.mark.parametrize(
    ("field", "value"),
    [("method", "ppo"), ("epochs", 0), ("torch_threads", 0), ("learning_rate", 0)],
)
def test_invalid_training_profile_is_rejected(tmp_path, field, value):
    values = {"profile_id": "bad", "dataset": "data.jsonl", "output": "model.pt", field: value}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_profile(path)
