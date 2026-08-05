from __future__ import annotations

import torch
import pytest

from shards_ai.ai import NeuralActionScorer, NeuralModelConfig, NeuralPlayer, RandomPlayer
from shards_ai.game import Game, GameRandom, GameRunner, PlayerId
from shards_ai.game.errors import InvalidGameStateError
from shards_ai.game.cards import CardInstance
from shards_ai.game.cards.definitions import VOID_ASSASSIN
from shards_ai.game.actions import BuyCard, RecruitMercenary


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


def test_mercenary_mode_bias_prefers_immediate_recruitment(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    _checkpoint(checkpoint)
    game = Game.new(seed=922)
    game.state.river[0] = CardInstance("mercenary", VOID_ASSASSIN)
    game.active.gems = VOID_ASSASSIN.cost
    game.apply(__import__("shards_ai.game.actions", fromlist=["PassPlayPhase"]).PassPlayPhase())

    class FixedScorer:
        def eval(self):
            return self

        def __call__(self, _observation, actions):
            return torch.zeros(len(actions))

    observation = game.neural_observation_for(game.active_player)
    legal_actions = game.legal_actions()
    player = NeuralPlayer(
        PlayerId.PLAYER_1,
        None,
        GameRandom(1),
        scorer=FixedScorer(),
        mercenary_mode_bias=1.0,
    )

    chosen = player.choose_action(observation, legal_actions)

    assert chosen == RecruitMercenary(0, "mercenary")
    assert BuyCard(0, "mercenary") in legal_actions


def test_deck_lean_bias_does_not_change_non_purchase_scores(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    _checkpoint(checkpoint)
    game = Game.new(seed=923)

    class FixedScorer:
        def eval(self):
            return self

        def __call__(self, _observation, actions):
            return torch.zeros(len(actions))

    player = NeuralPlayer(
        PlayerId.PLAYER_1,
        None,
        GameRandom(1),
        scorer=FixedScorer(),
        deck_lean_bias=1.0,
    )
    chosen = player.choose_action(
        game.neural_observation_for(game.active_player),
        game.legal_actions(),
    )

    assert chosen in game.legal_actions()
