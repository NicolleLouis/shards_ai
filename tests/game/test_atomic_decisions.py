from __future__ import annotations

from dataclasses import dataclass, field

from shards_ai.game import GameRunner, PlayerId
from shards_ai.game.actions import Action


@dataclass
class FirstLegalActionPlayer:
    """Minimal player used to audit the runner's atomic decision boundary."""

    decisions: list[tuple[object, tuple[Action, ...]]] = field(default_factory=list)

    def choose_action(self, observation, legal_actions):
        actions = tuple(legal_actions)
        self.decisions.append((observation, actions))
        return actions[0]


def test_runner_requests_one_legal_action_for_each_atomic_transition() -> None:
    players = {player_id: FirstLegalActionPlayer() for player_id in PlayerId}
    runner = GameRunner.random_duel(seed=208, max_turns=2)
    runner.players = players
    observed_decisions: list[tuple[tuple[Action, ...], Action]] = []

    runner.run(
        decision_observer=lambda _observation, actions, chosen, _player_id: observed_decisions.append(
            (tuple(actions), chosen)
        )
    )

    total_recorded = sum(len(player.decisions) for player in players.values())

    assert observed_decisions
    assert total_recorded == runner.actions_played
    assert len(observed_decisions) == runner.actions_played
    assert all(chosen in actions for actions, chosen in observed_decisions)
    assert all(actions for actions, _chosen in observed_decisions)


def test_runner_never_batches_multiple_actions_into_one_player_decision() -> None:
    players = {player_id: FirstLegalActionPlayer() for player_id in PlayerId}
    runner = GameRunner.random_duel(seed=209, max_turns=1)
    runner.players = players
    transitions: list[Action] = []

    runner.run(
        decision_observer=lambda _observation, _actions, chosen, _player_id: transitions.append(chosen)
    )

    assert transitions
    assert len(transitions) == runner.actions_played
    assert all(isinstance(action, Action) for action in transitions)
