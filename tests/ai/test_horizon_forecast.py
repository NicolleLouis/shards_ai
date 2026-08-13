from __future__ import annotations

import json

import pytest
import torch

from shards_ai.ai.horizon_forecast import (
    HORIZON_FEATURE_SET_BASELINE,
    HORIZON_FEATURE_SET_V1,
    HorizonFeatures,
    evaluate_model,
    horizon_class_for_remaining_turns,
    model_for_feature_set,
    project_baseline_dataset,
    split_for_game_id,
)


def test_horizon_vectors_have_stable_dimensions_and_normalization() -> None:
    features = HorizonFeatures(
        turn_number=10,
        active_health=25,
        opponent_health=50,
        active_mastery=3,
        opponent_mastery=6,
        active_owned_card_count=12,
        opponent_owned_card_count=15,
        active_faction_counts=(1, 2, 3, 4),
        opponent_faction_counts=(5, 6, 7, 8),
    )
    assert len(features.baseline_vector()) == 1
    assert len(features.v1_vector()) == 15
    assert features.baseline_vector() == (0.1,)
    assert features.v1_vector()[1:3] == (0.5, 1.0)


def test_feature_set_models_have_expected_input_sizes() -> None:
    assert model_for_feature_set(HORIZON_FEATURE_SET_BASELINE).network[0].in_features == 1
    assert model_for_feature_set(HORIZON_FEATURE_SET_V1).network[0].in_features == 15
    assert model_for_feature_set(HORIZON_FEATURE_SET_V1).network[-1].out_features == 7


def test_horizon_classes_cap_at_t6_plus() -> None:
    assert [horizon_class_for_remaining_turns(value) for value in range(7)] == list(range(7))
    assert horizon_class_for_remaining_turns(20) == 6
    with pytest.raises(ValueError):
        horizon_class_for_remaining_turns(-1)


def test_baseline_projection_keeps_targets_and_removes_enriched_features(tmp_path) -> None:
    source = tmp_path / "canonical.jsonl"
    destination = tmp_path / "baseline.jsonl"
    source.write_text(
        json.dumps(
            {
                "game_id": "game-1",
                "game_seed": 7,
                "decision_index": 2,
                "active_player": 1,
                "features": {
                    "turn_number": 4,
                    "active_health": 20,
                    "opponent_health": 30,
                    "active_mastery": 2,
                    "opponent_mastery": 5,
                    "active_owned_card_count": 10,
                    "opponent_owned_card_count": 11,
                    "active_faction_counts": [1, 2, 3, 4],
                    "opponent_faction_counts": [4, 3, 2, 1],
                },
                "target_horizon_class": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert project_baseline_dataset(source, destination) == 1
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["feature_set"] == HORIZON_FEATURE_SET_BASELINE
    assert record["features"] == {"turn_number": 4}
    assert record["target_horizon_class"] == 3


def test_game_split_is_reproducible() -> None:
    assert split_for_game_id("game-1", 12) == split_for_game_id("game-1", 12)
    assert {split_for_game_id(f"game-{index}", 12) for index in range(100)} == {
        "train", "validation", "test"
    }


def test_evaluation_reports_short_horizon_error() -> None:
    model = model_for_feature_set(HORIZON_FEATURE_SET_BASELINE)
    records = [
        {
            "features": {"turn_number": 1},
            "target_horizon_class": 0,
        },
        {
            "features": {"turn_number": 2},
            "target_horizon_class": 3,
        },
    ]
    metrics = evaluate_model(model, records, HORIZON_FEATURE_SET_BASELINE)
    assert metrics["records"] == 2
    assert torch.isfinite(model(torch.tensor([[0.01], [0.02]]))).all()
    assert "short_t0_t2_recall" in metrics
    assert len(metrics["confusion_matrix"]) == 7
