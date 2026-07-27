from __future__ import annotations

import json
from dataclasses import asdict

import torch

from shards_ai.ai import NeuralActionScorer, NeuralModelConfig, representation_for_action
from shards_ai.ai.neural_training import (
    chosen_action_loss,
    combined_imitation_loss,
    observation_from_dict,
    pairwise_ranking_loss,
    train_epoch,
)
from shards_ai.game import Game


def _decision_fixture() -> tuple[Game, object, list]:
    game = Game.new(seed=901)
    observation = game.neural_observation_for(game.active_player)
    representations = [representation_for_action(action, game.state) for action in game.legal_actions()]
    return game, observation, representations


def test_action_conditioned_model_scores_all_legal_actions() -> None:
    _game, observation, actions = _decision_fixture()
    model = NeuralActionScorer(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    scores = model(observation, actions)

    assert scores.shape == (len(actions),)
    assert torch.isfinite(scores).all()


def test_model_accepts_serialized_masked_observation() -> None:
    _game, observation, actions = _decision_fixture()
    restored = observation_from_dict(json.loads(json.dumps(asdict(observation))))
    model = NeuralActionScorer()

    assert model(restored, actions).shape == (len(actions),)


def test_ranking_loss_only_uses_strict_teacher_order() -> None:
    predicted = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss = pairwise_ranking_loss(predicted, torch.tensor([2.0, 1.0, 1.0]))

    assert loss.item() > 0
    loss.backward()
    assert predicted.grad is not None

    assert pairwise_ranking_loss(predicted.detach(), torch.tensor([1.0, 1.0, 1.0])).item() == 0


def test_combined_loss_can_train_one_json_record() -> None:
    game, observation, actions = _decision_fixture()
    model = NeuralActionScorer(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    record = {
        "observation": asdict(observation),
        "action_representations": [action.to_dict() for action in actions],
        "heuristic_scores": list(range(len(actions), 0, -1)),
        "chosen_action_index": 0,
    }

    metrics = train_epoch(model, [record], optimizer)

    assert metrics.records == 1
    assert metrics.mean_loss >= 0
    assert chosen_action_loss(torch.tensor([1.0, 0.0]), 0).item() > 0
    assert combined_imitation_loss(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0]), 0).item() >= 0


def test_single_action_decision_never_produces_nan_loss() -> None:
    predicted = torch.tensor([0.5], requires_grad=True)

    loss = combined_imitation_loss(predicted, torch.tensor([3.0]), 0)

    assert torch.isfinite(loss)
