from __future__ import annotations

from dataclasses import asdict

import torch

from shards_ai.ai import NeuralActionScorer, representation_for_action
from shards_ai.ai.neural_training import split_for_game_id
from shards_ai.game import Game


def test_game_split_is_deterministic_and_game_scoped() -> None:
    first = split_for_game_id("game-42", seed=12)
    assert first == split_for_game_id("game-42", seed=12)
    assert first in {"train", "validation", "test"}


def test_split_seed_changes_partition_function() -> None:
    assignments = {split_for_game_id(f"game-{index}", seed=seed) for index in range(100) for seed in (1, 2)}
    assert assignments == {"train", "validation", "test"}


def test_seed_training_rejects_invalid_thread_count() -> None:
    from shards_ai.ai.neural_training import seed_training

    try:
        seed_training(1, torch_threads=0)
    except ValueError as error:
        assert "torch_threads" in str(error)
    else:
        raise AssertionError("Expected invalid thread count to fail")


def test_evaluation_metrics_capture_imitation_and_ranking() -> None:
    game = Game.new(seed=902)
    observation = game.neural_observation_for(game.active_player)
    actions = [representation_for_action(action, game.state) for action in game.legal_actions()]
    record = {
        "observation": asdict(observation),
        "action_representations": [action.to_dict() for action in actions],
        "heuristic_scores": list(range(len(actions), 0, -1)),
        "chosen_action_index": 0,
    }
    model = NeuralActionScorer()

    from shards_ai.ai.neural_training import evaluate_epoch
    metrics = evaluate_epoch(model, [record])

    assert metrics.records == 1
    assert 0 <= metrics.top1_accuracy <= 1
    assert 0 <= metrics.pairwise_accuracy <= 1
    assert metrics.pairwise_pairs > 0


def test_train_epoch_accepts_positive_record_weights() -> None:
    from shards_ai.ai.neural_training import train_epoch

    game = Game.new(seed=903)
    observation = game.neural_observation_for(game.active_player)
    actions = [representation_for_action(action, game.state) for action in game.legal_actions()]
    record = {
        "observation": asdict(observation),
        "action_representations": [action.to_dict() for action in actions],
        "heuristic_scores": list(range(len(actions), 0, -1)),
        "chosen_action_index": 0,
        "chosen_action": {"action_type": "buy_card"},
    }
    model = NeuralActionScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = train_epoch(model, [record], optimizer, record_weight=lambda _: 2.0)

    assert metrics.records == 1
    assert metrics.mean_loss > 0


def test_targeted_mercenary_record_requires_both_routes_for_same_card() -> None:
    from shards_ai.ai.neural_training import is_targeted_mercenary_record

    targeted = {
        "action_representations": [
            {"action_type": "buy_card", "card_definition_id": "void_assassin"},
            {"action_type": "recruit_mercenary", "card_definition_id": "void_assassin"},
        ]
    }
    unrelated = {
        "action_representations": [
            {"action_type": "buy_card", "card_definition_id": "void_assassin"},
            {"action_type": "recruit_mercenary", "card_definition_id": "apotre_des_ombres"},
        ]
    }

    assert is_targeted_mercenary_record(targeted)
    assert not is_targeted_mercenary_record(unrelated)
