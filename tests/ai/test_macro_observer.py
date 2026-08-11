from __future__ import annotations

from shards_ai.ai import MacroNeuralPlayer
from shards_ai.game import Game, GameRunner
from shards_ai.ai.random_player import RandomPlayer
from shards_ai.game.random import GameRandom


def test_runner_exposes_macro_decision_payload_once_per_branch_choice() -> None:
    game = Game.new(seed=1401)
    macro_id = game.active_player
    opponent_id = macro_id.opponent
    macro = MacroNeuralPlayer(macro_id, game)
    opponent = RandomPlayer(opponent_id, GameRandom(1402))
    payloads = []

    runner = GameRunner(game, {macro_id: macro, opponent_id: opponent}, max_turns=1)
    runner.run(macro_decision_observer=lambda payload, player_id: payloads.append((payload, player_id)))

    assert payloads
    payload, player_id = payloads[0]
    assert player_id is macro_id
    assert payload.candidate_representations
    assert payload.selected_atomic_trace
