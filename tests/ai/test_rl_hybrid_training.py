from dataclasses import replace

import torch
import pytest

from shards_ai.ai.neural_training_profiles import load_training_profile
from shards_ai.ai.rl_training import (
    NeuralActorCritic,
    PPOTrainingAcquisitionPolicy,
    collect_hybrid_acquisition_rollout,
)
from shards_ai.game import Game, GameRandom, PlayerId
from shards_ai.game.actions import PassPlayPhase


PROFILE = "configs/neural_training_profiles/candidates/ppo-deckbuilding-hybrid-v003.yaml"


def _model():
    checkpoint = torch.load(
        "configs/neural_profiles/v006.pt", map_location="cpu", weights_only=False,
    )
    return NeuralActorCritic.from_checkpoint(checkpoint)


def test_acquisition_policy_rejects_non_acquisition_candidates():
    game = Game.new(seed=12001, rng=GameRandom(12001))
    policy = PPOTrainingAcquisitionPolicy(PlayerId.PLAYER_1, game, _model())

    with pytest.raises(ValueError, match="non-acquisition"):
        policy.choose_action(game.state, [PassPlayPhase()])


def test_hybrid_rollout_keeps_only_acquisition_transitions_and_terminal_reward():
    profile = load_training_profile(PROFILE)
    profile = replace(profile, opponents={"neural:v005": 1.0})
    rollout = collect_hybrid_acquisition_rollout(
        _model(), profile, start_game_index=0, games=1,
    )

    assert rollout.games == 1
    assert rollout.transitions
    assert all(
        representation.root_action is not None
        and representation.root_action.action_type in {
            "buy_card", "recruit_mercenary", "recruit_free_card", "stop_buying",
        }
        for transition in rollout.transitions
        for representation in transition.legal_action_representations
    )
    assert all(transition.reward == 0.0 for transition in rollout.transitions[:-1])
    assert rollout.transitions[-1].done is True
    assert rollout.transitions[-1].reward in {-1.0, 0.0, 1.0}
