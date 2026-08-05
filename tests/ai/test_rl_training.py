from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest

from shards_ai.ai import (
    NeuralActorCritic,
    NeuralModelConfig,
    PPOUpdateMetrics,
    RolloutTransition,
    collect_rollout,
    choose_opponent,
    evaluate_greedy_model,
    gae_returns,
    is_monotonic_evaluation_improvement,
    load_training_profile,
    ppo_update,
    terminal_reward,
    weighted_evaluation_score,
)
from shards_ai.game import Game, GameRandom, GameStatus, PlayerId
from shards_ai.ai.action_representation import ActionRepresentation


def _transition(game, *, episode_id, reward=0.0, value=0.0, done=False):
    observation = game.neural_observation_for(game.active_player)
    return RolloutTransition(
        episode_id=episode_id,
        game_seed=game.state.seed,
        opponent_id="random",
        neural_player_id=PlayerId.PLAYER_1,
        turn_number=observation.turn_number,
        observation=observation,
        legal_action_representations=(ActionRepresentation("pass_play_phase", "play"),),
        chosen_action_index=0,
        old_log_probability=0.0,
        value_estimate=value,
        reward=reward,
        done=done,
    )


def test_terminal_reward_is_only_outcome_from_learner_perspective():
    assert terminal_reward(SimpleNamespace(status=GameStatus.RUNNING, winner=None), PlayerId.PLAYER_1) == 0.0
    assert terminal_reward(SimpleNamespace(status=GameStatus.DRAW, winner=None), PlayerId.PLAYER_1) == 0.0
    assert terminal_reward(SimpleNamespace(status=GameStatus.FINISHED, winner=PlayerId.PLAYER_1), PlayerId.PLAYER_1) == 1.0
    assert terminal_reward(SimpleNamespace(status=GameStatus.FINISHED, winner=PlayerId.PLAYER_1), PlayerId.PLAYER_2) == -1.0


def test_gae_does_not_carry_credit_across_episodes():
    game = Game.new(seed=1001)
    transitions = (
        _transition(game, episode_id=0, reward=1.0, done=True),
        _transition(game, episode_id=1),
        _transition(game, episode_id=1, reward=-1.0, done=True),
    )

    advantages, returns = gae_returns(transitions, gamma=0.9, gae_lambda=0.95)

    assert torch.allclose(advantages, torch.tensor([1.0, -0.855, -1.0]), atol=1e-6)
    assert torch.allclose(returns, advantages, atol=1e-6)


def test_opponent_choice_is_reproducible_and_weighted_names_are_preserved():
    weights = {"random": 1.0, "v007": 2.0, "v008": 3.0}
    first = [choose_opponent(GameRandom(7), weights) for _ in range(3)]
    second = [choose_opponent(GameRandom(7), weights) for _ in range(3)]

    assert first == second
    assert set(first) <= set(weights)


def test_opponent_choice_rejects_a_mix_without_positive_weight():
    with pytest.raises(ValueError, match="positive weight"):
        choose_opponent(GameRandom(8), {"random": 0.0, "v007": -1.0})


def test_ppo_update_rejects_an_empty_rollout():
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="empty rollout"):
        ppo_update(
            model,
            optimizer,
            [],
            optimization_epochs=1,
            minibatch_size=1,
            gamma=0.995,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coefficient=0.5,
            entropy_coefficient=0.01,
        )


def test_collect_rollout_records_one_terminal_transition_per_episode_reproducibly():
    profile = replace(
        load_training_profile("configs/neural_training_profiles/candidates/v002.yaml"),
        max_turns=10,
        max_actions=100,
    )
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    torch.manual_seed(9)
    first = collect_rollout(model, profile, start_game_index=0, games=1)
    torch.manual_seed(9)
    second = collect_rollout(model, profile, start_game_index=0, games=1)

    assert first.games == second.games == 1
    assert first.transitions_by_episode == second.transitions_by_episode
    assert first.games_by_opponent == second.games_by_opponent
    assert first.outcomes_by_opponent == second.outcomes_by_opponent
    assert first.transitions[-1].done is True
    assert all(not transition.done for transition in first.transitions[:-1])
    assert [transition.chosen_action_index for transition in first.transitions] == [
        transition.chosen_action_index for transition in second.transitions
    ]


def test_parallel_rollout_preserves_sequential_game_order_and_actions():
    profile = replace(
        load_training_profile("configs/neural_training_profiles/candidates/v002.yaml"),
        max_turns=10,
        max_actions=100,
    )
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    sequential = collect_rollout(model, profile, start_game_index=0, games=2, workers=1)
    parallel = collect_rollout(model, profile, start_game_index=0, games=2, workers=2)

    assert parallel.transitions_by_episode == sequential.transitions_by_episode
    assert parallel.games_by_opponent == sequential.games_by_opponent
    assert parallel.outcomes_by_opponent == sequential.outcomes_by_opponent
    assert [transition.game_seed for transition in parallel.transitions] == [
        transition.game_seed for transition in sequential.transitions
    ]
    assert [transition.chosen_action_index for transition in parallel.transitions] == [
        transition.chosen_action_index for transition in sequential.transitions
    ]


def test_collect_rollout_rejects_non_positive_worker_count():
    profile = load_training_profile("configs/neural_training_profiles/candidates/v002.yaml")
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    with pytest.raises(ValueError, match="workers must be positive"):
        collect_rollout(model, profile, start_game_index=0, games=1, workers=0)


def test_actor_critic_reuses_scorer_shape_and_has_value_head():
    game = Game.new(seed=1002)
    observation = game.neural_observation_for(game.active_player)
    from shards_ai.ai import representation_for_neural_action

    actions = game.legal_actions()
    representations = [representation_for_neural_action(action, observation) for action in actions]
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    logits = model.policy_logits(observation, representations)
    value = model.value(observation)

    assert logits.shape == (len(actions),)
    assert value.shape == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_contextual_actor_critic_scores_the_candidate_set():
    game = Game.new(seed=1004)
    observation = game.neural_observation_for(game.active_player)
    from shards_ai.ai import representation_for_neural_action

    representations = [
        representation_for_neural_action(action, observation)
        for action in game.legal_actions()
    ]
    model = NeuralActorCritic(
        NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16, candidate_context_dim=8),
        architecture="global_candidate_context",
    )

    logits, value = model.evaluate(observation, representations)

    assert logits.shape == (len(representations),)
    assert value.shape == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_ppo_update_returns_finite_metrics_for_one_terminal_transition():
    game = Game.new(seed=1003)
    observation = game.neural_observation_for(game.active_player)
    from shards_ai.ai import representation_for_neural_action

    actions = game.legal_actions()
    transition = RolloutTransition(
        episode_id=0,
        game_seed=game.state.seed,
        opponent_id="random",
        neural_player_id=game.active_player,
        turn_number=observation.turn_number,
        observation=observation,
        legal_action_representations=tuple(
            representation_for_neural_action(action, observation) for action in actions
        ),
        chosen_action_index=0,
        old_log_probability=0.0,
        value_estimate=0.0,
        reward=1.0,
        done=True,
    )
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = ppo_update(
        model,
        optimizer,
        [transition],
        optimization_epochs=1,
        minibatch_size=1,
        gamma=0.995,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    assert isinstance(metrics, PPOUpdateMetrics)
    assert metrics.transitions == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in (
        metrics.policy_loss, metrics.value_loss, metrics.entropy, metrics.approx_kl,
    ))


def test_ppo_update_reports_reference_kl_when_reference_policy_is_provided():
    game = Game.new(seed=1004)
    observation = game.neural_observation_for(game.active_player)
    from shards_ai.ai import representation_for_neural_action

    actions = game.legal_actions()
    transition = RolloutTransition(
        episode_id=0,
        game_seed=game.state.seed,
        opponent_id="random",
        neural_player_id=game.active_player,
        turn_number=observation.turn_number,
        observation=observation,
        legal_action_representations=tuple(
            representation_for_neural_action(action, observation) for action in actions
        ),
        chosen_action_index=0,
        old_log_probability=0.0,
        value_estimate=0.0,
        reward=1.0,
        done=True,
    )
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    reference = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics = ppo_update(
        model,
        optimizer,
        [transition],
        optimization_epochs=1,
        minibatch_size=1,
        gamma=0.995,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coefficient=0.5,
        entropy_coefficient=0.001,
        reference_model=reference,
        reference_kl_coefficient=0.02,
    )

    assert torch.isfinite(torch.tensor(metrics.reference_kl))
    assert metrics.reference_kl >= 0.0


def test_greedy_evaluation_returns_a_score_for_each_opponent():
    profile = replace(
        load_training_profile("configs/neural_training_profiles/candidates/v002.yaml"),
        evaluation_games=1,
        max_turns=10,
        max_actions=1000,
    )
    model = NeuralActorCritic(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    evaluation = evaluate_greedy_model(model, profile)

    assert set(evaluation["by_opponent"]) == {"random", "v007", "v008"}
    assert 0.0 <= evaluation["score"] <= 1.0


def test_evaluation_score_uses_profile_weights_and_favors_v008():
    evaluation = {
        "by_opponent": {
            "random": {"win_rate": 1.0},
            "v007": {"win_rate": 0.5},
            "v008": {"win_rate": 0.0},
        }
    }

    assert weighted_evaluation_score(evaluation, {"random": 0.2, "v007": 0.3, "v008": 0.5}) == pytest.approx(0.35)


def test_evaluation_selection_rejects_regression_against_any_opponent():
    incumbent = {
        "by_opponent": {
            "random": {"win_rate": 0.5},
            "v007": {"win_rate": 0.5},
            "v008": {"win_rate": 0.5},
        }
    }
    candidate = {
        "by_opponent": {
            "random": {"win_rate": 1.0},
            "v007": {"win_rate": 1.0},
            "v008": {"win_rate": 0.25},
        }
    }

    assert not is_monotonic_evaluation_improvement(
        candidate,
        incumbent,
        {"random": 0.2, "v007": 0.3, "v008": 0.5},
    )


def test_evaluation_selection_allows_one_reference_loss_with_better_weighted_score():
    incumbent = {
        "by_opponent": {
            "random": {"win_rate": 0.890625},
            "v007": {"win_rate": 0.515625},
            "v008": {"win_rate": 0.21875},
        }
    }
    candidate = {
        "by_opponent": {
            "random": {"win_rate": 0.90625},
            "v007": {"win_rate": 0.5},
            "v008": {"win_rate": 0.234375},
        }
    }

    assert is_monotonic_evaluation_improvement(
        candidate,
        incumbent,
        {"random": 0.2, "v007": 0.3, "v008": 0.5},
        tolerated_opponents=("random", "v007"),
        tolerance_rate=1 / 64,
    )


def test_evaluation_selection_never_tolerates_a_v008_regression():
    incumbent = {
        "by_opponent": {
            "random": {"win_rate": 0.890625},
            "v007": {"win_rate": 0.515625},
            "v008": {"win_rate": 0.21875},
        }
    }
    candidate = {
        "by_opponent": {
            "random": {"win_rate": 0.90625},
            "v007": {"win_rate": 0.515625},
            "v008": {"win_rate": 0.203125},
        }
    }

    assert not is_monotonic_evaluation_improvement(
        candidate,
        incumbent,
        {"random": 0.2, "v007": 0.3, "v008": 0.5},
        tolerated_opponents=("random", "v007"),
        tolerance_rate=1 / 64,
    )
