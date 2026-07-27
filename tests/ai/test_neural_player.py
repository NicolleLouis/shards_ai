from __future__ import annotations

import torch
import pytest

from shards_ai.ai import NeuralActionScorer, NeuralModelConfig, NeuralPlayer, RandomPlayer
from shards_ai.game import Game, GameRandom, GameRunner, PlayerId
from shards_ai.game.errors import InvalidGameStateError


def _checkpoint(path) -> None:
    model = NeuralActionScorer(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "card_embedding_dim": 16,
            "card_id_embedding_dim": 12,
            "semantic_hidden_dim": 48,
            "state_hidden_dim": 32,
            "action_hidden_dim": 16,
            "scorer_hidden_dim": 96,
        },
        "card_ids": model.card_ids,
    }, path)


def test_neural_player_chooses_one_legal_action(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    _checkpoint(checkpoint)
    game = Game.new(seed=920)
    player = NeuralPlayer(PlayerId.PLAYER_1, checkpoint, GameRandom(1))

    action = player.choose_action(
        game.neural_observation_for(game.active_player),
        game.legal_actions(),
    )

    assert action in game.legal_actions()


def test_runner_provides_masked_observation_to_neural_player(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    _checkpoint(checkpoint)
    game = Game.new(seed=921)
    neural = NeuralPlayer(PlayerId.PLAYER_1, checkpoint, GameRandom(1))
    random = RandomPlayer(PlayerId.PLAYER_2, GameRandom(2))
    runner = GameRunner(game, {PlayerId.PLAYER_1: neural, PlayerId.PLAYER_2: random}, max_actions=1)

    with pytest.raises(InvalidGameStateError, match="max_actions=1"):
        runner.run()

    assert runner.actions_played == 1
