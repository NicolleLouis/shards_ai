from __future__ import annotations

from dataclasses import replace

import pytest

from shards_ai.ai import MacroNeuralPlayer, PlayTurnSolver
from shards_ai.game import Game, Phase
from shards_ai.game.actions import BanishCard, GainMastery, PassPlayPhase, PlayCard, SkipBanish
from shards_ai.game.cards.definitions import (
    APOTRE_DES_OMBRES,
    BLASTER,
    CRYSTAL,
    DRONES_MINIERS,
    ECLAIREUR_SPECTRAL,
    ERMITE_FONGIQUE,
    SHARD_REACTOR,
)
from shards_ai.game.cards import CardInstance


def test_fixed_cards_are_reduced_to_one_canonical_prefix() -> None:
    game = Game.new(seed=1301, card_definition=CRYSTAL)

    resolution = PlayTurnSolver().resolve(game)

    assert len(resolution.automatic_prefix) == game.STARTING_HAND_SIZE
    assert all(isinstance(action, PlayCard) for action in resolution.automatic_prefix)
    assert len(resolution.candidates) == 2
    assert any(candidate.atomic_trace == (PassPlayPhase(),) for candidate in resolution.candidates)
    assert any(candidate.summary.terminal_kind == "strategic_choice" for candidate in resolution.candidates)


def test_equivalent_cards_in_hand_share_one_macro_candidate() -> None:
    game = Game.new(seed=1310)
    game.active.hand = [
        CardInstance("crystal-1", CRYSTAL),
        CardInstance("crystal-2", CRYSTAL),
        CardInstance("blaster-1", BLASTER),
        CardInstance("banisher", APOTRE_DES_OMBRES),
    ]
    resolution = PlayTurnSolver().resolve(game)

    assert not any(
        candidate.atomic_trace[0] == PlayCard("crystal-2")
        for candidate in resolution.candidates
    )
    crystal = next(
        candidate
        for candidate in resolution.candidates
        if candidate.atomic_trace[0] == PlayCard("crystal-1")
    )
    assert crystal.physical_variant_count == 2
    assert crystal.representation.physical_variant_count == 2
    assert len(resolution.candidates) == 4


def test_equivalent_banish_targets_share_one_atomic_candidate() -> None:
    game = Game.new(seed=1311)
    game.active.hand = [
        CardInstance("crystal-1", CRYSTAL),
        CardInstance("crystal-2", CRYSTAL),
        CardInstance("blaster-1", BLASTER),
    ]
    game.active.pending_banishes = 1
    resolution = PlayTurnSolver().resolve(game)

    roots = [candidate.atomic_trace[0] for candidate in resolution.candidates]
    assert BanishCard("crystal-1") in roots
    assert BanishCard("crystal-2") not in roots
    assert SkipBanish() in roots
    crystal = next(candidate for candidate in resolution.candidates if candidate.atomic_trace[0] == BanishCard("crystal-1"))
    assert crystal.physical_variant_count == 2


def test_macro_player_replays_the_physical_representative_of_a_group() -> None:
    game = Game.new(seed=1312)
    game.active.hand = [
        CardInstance("crystal-1", CRYSTAL),
        CardInstance("crystal-2", CRYSTAL),
        CardInstance("blaster-1", BLASTER),
        CardInstance("banisher", APOTRE_DES_OMBRES),
    ]
    seen = []

    def choose_crystal(_game, _observation, candidates) -> int:
        seen.extend(candidates)
        return next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.atomic_trace[0] == PlayCard("crystal-1")
        )

    player = MacroNeuralPlayer(game.active_player, game, candidate_scorer=choose_crystal)
    action = player.choose_action(game.neural_observation_for(game.active_player), game.legal_actions())

    assert action == PlayCard("crystal-1")
    assert any(candidate.physical_variant_count == 2 for candidate in seen)
    assert player.pop_last_macro_decision().physical_variant_count == 2


def test_conditional_cards_remain_neural_candidates() -> None:
    game = Game.new(seed=1302, card_definition=SHARD_REACTOR)

    resolution = PlayTurnSolver().resolve(game)

    assert resolution.automatic_prefix == ()
    assert resolution.candidates
    assert all(candidate.atomic_trace for candidate in resolution.candidates)
    assert any(
        isinstance(candidate.atomic_trace[0], PlayCard)
        for candidate in resolution.candidates
    )


def test_highest_mastery_branch_is_canonical_when_already_active() -> None:
    game = Game.new(seed=1306, card_definition=SHARD_REACTOR)
    game.active.mastery = 15
    card_action = next(action for action in game.legal_actions() if isinstance(action, PlayCard))

    descriptor = __import__("shards_ai.ai", fromlist=["dependency_for_action"]).dependency_for_action(
        game, card_action
    )

    assert descriptor.canonicalizable


def test_active_echo_is_not_canonicalized_across_a_possible_draw() -> None:
    game = Game.new(seed=1307)
    game.active.hand = [
        CardInstance("echo-card", ECLAIREUR_SPECTRAL),
        CardInstance("draw-card", DRONES_MINIERS),
    ]
    game.active.discard_pile = [CardInstance("discard-spectra", ECLAIREUR_SPECTRAL)]

    resolution = PlayTurnSolver().resolve(game)

    assert resolution.automatic_prefix == ()
    assert any(
        candidate.atomic_trace[0] == PlayCard("echo-card")
        for candidate in resolution.candidates
    )
    assert any(
        candidate.atomic_trace[0] == PlayCard("draw-card")
        for candidate in resolution.candidates
    )


def test_mastery_producing_card_is_not_canonicalized() -> None:
    game = Game.new(seed=1305, card_definition=ERMITE_FONGIQUE)
    game.active.hand = [game.active.hand[0]]

    resolution = PlayTurnSolver().resolve(game)

    assert resolution.automatic_prefix == ()
    assert any(
        isinstance(candidate.atomic_trace[0], PlayCard)
        for candidate in resolution.candidates
    )


def test_solver_budgets_are_architecture_constants() -> None:
    with pytest.raises(ValueError, match="fixed architecture constants"):
        PlayTurnSolver(max_expansions=1)


def test_macro_player_replays_prefix_and_branch_atomically() -> None:
    game = Game.new(seed=1303, card_definition=CRYSTAL)
    player = MacroNeuralPlayer(game.active_player, game)
    applied = []

    while game.state.phase is Phase.PLAY:
        legal_actions = game.legal_actions()
        action = player.choose_action(
            game.neural_observation_for(game.active_player),
            legal_actions,
        )
        assert action in legal_actions
        applied.append(action)
        game.apply(action)

    assert len(applied) == game.STARTING_HAND_SIZE + 2
    assert all(isinstance(action, (GainMastery, PlayCard, PassPlayPhase)) for action in applied)
    assert player.atomic_replays == len(applied)
    assert player.macro_decisions == 1


def test_macro_scorer_selects_a_candidate_and_replays_its_trace() -> None:
    game = Game.new(seed=1304, card_definition=SHARD_REACTOR)
    selected = []

    def choose_last(_game, _observation, candidates) -> int:
        selected.append(tuple(candidate.representation for candidate in candidates))
        return len(candidates) - 1

    player = MacroNeuralPlayer(
        game.active_player,
        game,
        candidate_scorer=choose_last,
    )
    first_legal = game.legal_actions()
    first = player.choose_action(
        game.neural_observation_for(game.active_player),
        first_legal,
    )

    assert selected
    assert first in first_legal
    assert player.macro_decisions == 1


def test_singleton_solver_resolution_replays_without_scoring() -> None:
    game = Game.new(seed=1305, card_definition=SHARD_REACTOR)
    resolution = PlayTurnSolver().resolve(game)
    assert resolution.candidates
    singleton = replace(resolution, candidates=(resolution.candidates[0],))
    calls = []

    class StubSolver:
        def resolve(self, _game):
            return singleton

    player = MacroNeuralPlayer(
        game.active_player,
        game,
        solver=StubSolver(),
        candidate_scorer=lambda *_args: calls.append(True) or 0,
    )
    action = player.choose_action(
        game.neural_observation_for(game.active_player),
        game.legal_actions(),
    )

    assert action in game.legal_actions()
    assert calls == []
    assert player.macro_decisions == 0
    assert player.pop_last_macro_decision() is None


def test_non_play_decisions_use_the_unified_atomic_candidate_contract() -> None:
    game = Game.new(seed=1308)
    game.state.phase = Phase.BUY
    game.active.gems = 10
    seen = []

    def choose_last(_game, _observation, candidates) -> int:
        seen.extend(candidate.representation for candidate in candidates)
        return len(candidates) - 1

    player = MacroNeuralPlayer(game.active_player, game, candidate_scorer=choose_last)
    action = player.choose_action(
        game.neural_observation_for(game.active_player),
        game.legal_actions(),
    )

    assert action in game.legal_actions()
    assert len(seen) == len(game.legal_actions())
    assert all(candidate.decision_kind == "atomic" for candidate in seen)
    assert all(candidate.atomic_action_count == 1 for candidate in seen)
    assert player.atomic_decisions == 1


def test_solver_budget_boundary_uses_unified_atomic_scoring() -> None:
    game = Game.new(seed=1309)
    calls = []

    class BudgetBoundarySolver:
        def resolve(self, _game):
            from shards_ai.ai.play_turn_solver import PlayTurnResolution

            return PlayTurnResolution((), (), game, 0, 0, "test")

    def choose_first(_game, _observation, candidates) -> int:
        calls.append(candidates)
        return 0

    player = MacroNeuralPlayer(
        game.active_player,
        game,
        solver=BudgetBoundarySolver(),
        candidate_scorer=choose_first,
    )
    action = player.choose_action(
        game.neural_observation_for(game.active_player),
        game.legal_actions(),
    )

    assert action in game.legal_actions()
    assert calls
    assert all(candidate.representation.decision_kind == "atomic" for candidate in calls[0])
    assert player.atomic_decisions == 1
