from dataclasses import replace

import pytest
import torch

from shards_ai.ai import (
    NeuralActorCritic,
    collect_rollout,
    load_training_profile,
)
from shards_ai.ai.macro_player import _atomic_candidate
from shards_ai.ai.play_turn_solver import PlayTurnSolver, macro_representations_v4
from shards_ai.game import Game


MACRO_ARCHITECTURE = "structured_semantic_v5_macro_tactical_action_v1"


def _macro_model() -> NeuralActorCritic:
    checkpoint = torch.load(
        "configs/neural_profiles/v005.pt", map_location="cpu", weights_only=False,
    )
    return NeuralActorCritic.from_checkpoint(checkpoint)


def test_ppo_macro_actor_loads_v005_and_scores_variable_macro_candidates():
    model = _macro_model()
    assert model.architecture == MACRO_ARCHITECTURE

    game = Game.new(seed=8123)
    observation = game.neural_observation_for(game.active_player)
    resolution = PlayTurnSolver().resolve(game)
    macro_candidates = macro_representations_v4(observation, resolution.candidates)
    atomic_candidates = tuple(
        _atomic_candidate(observation, action, 4) for action in game.legal_actions()
    )

    macro_logits, macro_value = model.evaluate(observation, macro_candidates)
    atomic_logits, atomic_value = model.evaluate(
        observation, tuple(candidate.representation for candidate in atomic_candidates)
    )
    assert torch.isfinite(macro_logits).all()
    assert torch.isfinite(atomic_logits).all()
    assert torch.isfinite(macro_value).all()
    assert torch.isfinite(atomic_value).all()


def test_ppo_macro_actor_preserves_v005_macro_weights_when_loading_scorer_checkpoint():
    checkpoint = torch.load(
        "configs/neural_profiles/v005.pt", map_location="cpu", weights_only=False,
    )
    model = NeuralActorCritic.from_checkpoint(checkpoint)
    for key, expected in checkpoint["model_state_dict"].items():
        actual = model.inference_state_dict()[key]
        assert torch.equal(actual, expected), key


def test_macro_rollout_has_one_terminal_reward_per_episode_and_no_replay_transitions():
    profile = replace(
        load_training_profile(
            "configs/neural_training_profiles/candidates/ppo-v4-macro-play-turn-solver.yaml"
        ),
        max_turns=10,
        max_actions=1000,
    )
    rollout = collect_rollout(_macro_model(), profile, start_game_index=0, games=1)

    assert rollout.games == 1
    assert rollout.transitions
    assert all(transition.reward == 0 for transition in rollout.transitions[:-1])
    assert rollout.transitions[-1].done is True
    assert all(
        type(representation).__name__ == "MacroActionRepresentation"
        for transition in rollout.transitions
        for representation in transition.legal_action_representations
    )


def test_macro_profile_rejects_reward_shaping():
    profile = replace(
        load_training_profile(
            "configs/neural_training_profiles/candidates/ppo-v4-macro-play-turn-solver.yaml"
        ),
        reward_shaping={"enabled": True},
    )
    with pytest.raises(ValueError, match="Reward shaping"):
        collect_rollout(_macro_model(), profile, start_game_index=0, games=1)


def test_macro_ppo_profile_matches_the_promotion_panel_and_validation_budget():
    profile = load_training_profile(
        "configs/neural_training_profiles/candidates/ppo-v4-macro-play-turn-solver.yaml"
    )

    assert profile.evaluation_games == 100
    assert profile.evaluation_interval_games == 700
    assert profile.opponents == {
        "v007": 1.5,
        "v008": 2.0,
        "neural:v001": 0.5,
        "neural:v002": 0.5,
        "neural:v003": 0.25,
        "neural:v004": 0.25,
        "neural:v005": 1.0,
    }
    assert sum(profile.opponents.values()) == pytest.approx(6.0)
