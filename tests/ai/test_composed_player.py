from dataclasses import dataclass

import pytest

from shards_ai.ai import (
    AlgorithmicPlayPolicy,
    DecisionDiagnostic,
    DeterministicBanishPolicy,
    HybridPlayer,
    RandomPlayer,
    load_hybrid_profile,
)
from shards_ai.game import CardInstance, Game, GameRandom, GameRunner, PlayerId
from shards_ai.game.actions import BanishCard, BuyCard, PassPlayPhase, PlayCard, SkipBanish
from shards_ai.game.cards.definitions import BLASTER, CRYSTAL, INFINITY_SHARD
from shards_ai.game.cards import CardDefinition, Effect, EffectStep, Operation
from shards_ai.game.enums import Faction
from shards_ai.game.enums import Phase
from shards_ai.game.errors import InvalidActionError


@dataclass
class StubPolicy:
    policy_id: str
    action: object
    reason: str = "stub"

    def choose_action(self, observation, legal_actions):
        return self.action, self.reason


def _game() -> Game:
    game = Game.new(seed=8801)
    game.state.active_player = PlayerId.PLAYER_1
    return game


def test_banish_prefers_blaster_in_hand_or_discard() -> None:
    game = _game()
    player = game.state.players[PlayerId.PLAYER_1]
    player.hand = [CardInstance("hand-blaster", BLASTER)]
    player.discard_pile = [CardInstance("discard-crystal", CRYSTAL)]

    action, reason = DeterministicBanishPolicy().choose_action(
        game.state,
        [BanishCard("discard-crystal"), BanishCard("hand-blaster"), SkipBanish()],
    )

    assert action == BanishCard("hand-blaster")
    assert reason == "preferred_blaster"


def test_banish_prefers_discard_crystal_when_no_blaster_exists() -> None:
    game = _game()
    player = game.state.players[PlayerId.PLAYER_1]
    player.hand = [CardInstance("hand-crystal", CRYSTAL)]
    player.discard_pile = [CardInstance("discard-crystal", CRYSTAL)]

    action, reason = DeterministicBanishPolicy().choose_action(
        game.state,
        [BanishCard("hand-crystal"), BanishCard("discard-crystal"), SkipBanish()],
    )

    assert action == BanishCard("discard-crystal")
    assert reason == "preferred_discard_crystal"


def test_banish_skips_when_no_preferred_target_exists() -> None:
    game = _game()
    player = game.state.players[PlayerId.PLAYER_1]
    player.hand = [CardInstance("hand-shard", INFINITY_SHARD)]

    action, reason = DeterministicBanishPolicy().choose_action(
        game.state,
        [BanishCard("hand-shard"), SkipBanish()],
    )

    assert action == SkipBanish()
    assert reason == "no_preferred_banish_target"


@pytest.mark.parametrize(
    ("actions", "expected_family", "expected_policy"),
    [
        ([BuyCard(0, "buy")], "acquisition", "neural_stub"),
        ([BanishCard("banish"), SkipBanish()], "banish", "banish_stub"),
        ([PassPlayPhase()], "play", "play_stub"),
    ],
)
def test_hybrid_player_routes_each_decision_family(
    actions, expected_family, expected_policy
) -> None:
    game = _game()
    player = HybridPlayer(
        PlayerId.PLAYER_1,
        game,
        acquisition_policy=StubPolicy("neural_stub", actions[0]),
        play_policy=StubPolicy("play_stub", actions[0]),
        banish_policy=StubPolicy("banish_stub", actions[0]),
    )

    chosen = player.choose_action(game.state, actions)

    assert chosen == actions[0]
    assert player.last_decision == DecisionDiagnostic(
        policy_id=expected_policy,
        decision_family=expected_family,
        action_type=type(actions[0]).__name__,
        reason="stub",
    )


def test_banish_has_priority_over_acquisition_when_both_are_present() -> None:
    game = _game()
    player = HybridPlayer(
        PlayerId.PLAYER_1,
        game,
        acquisition_policy=StubPolicy("neural_stub", BuyCard(0, "buy")),
        play_policy=StubPolicy("play_stub", PassPlayPhase()),
        banish_policy=StubPolicy("banish_stub", SkipBanish()),
    )

    chosen = player.choose_action(
        game.state,
        [BuyCard(0, "buy"), BanishCard("banish"), SkipBanish()],
    )

    assert chosen == SkipBanish()
    assert player.last_decision is not None
    assert player.last_decision.decision_family == "banish"


def test_hybrid_player_rejects_an_illegal_policy_result() -> None:
    game = _game()
    player = HybridPlayer(
        PlayerId.PLAYER_1,
        game,
        acquisition_policy=StubPolicy("neural_stub", PassPlayPhase()),
        play_policy=StubPolicy("play_stub", PassPlayPhase()),
    )

    with pytest.raises(InvalidActionError, match="returned illegal action"):
        player.choose_action(game.state, [BuyCard(0, "buy")])


def test_versioned_hybrid_profile_is_replayable_and_explicit() -> None:
    profile = load_hybrid_profile("hybrid-v001")

    assert profile.profile_id == "hybrid-v001"
    assert profile.acquisition_policy_id == "neural_v006"
    assert profile.play_policy_id == "heuristic_v008"
    assert profile.banish_policy_id == "deterministic_blaster_crystal"
    assert profile.acquisition_checkpoint.as_posix().endswith(
        "configs/neural_profiles/v006.pt"
    )
    assert profile.play_profile.as_posix().endswith(
        "configs/heuristic_profiles/v008.yaml"
    )
    assert len(profile.fingerprint) == 64


def test_algorithmic_hybrid_version_references_independent_component_versions() -> None:
    profile = load_hybrid_profile("hybrid-v002")

    assert profile.parent_profile_id == "hybrid-v001"
    assert profile.play_policy_id == "algorithmic_play_v001"
    assert profile.banish_policy_id == "deterministic_blaster_crystal_v001"
    assert profile.acquisition_policy_profile is not None
    assert profile.play_policy_profile is not None
    assert profile.banish_policy_profile is not None
    assert profile.play_policy_profile.as_posix().endswith(
        "configs/player_policies/play/v001.yaml"
    )


def test_boundary_gain_mastery_hybrid_profile_is_explicit() -> None:
    profile = load_hybrid_profile("hybrid-v003")

    assert profile.parent_profile_id == "hybrid-v001"
    assert profile.capability_profile_id == "boundary_gain_mastery_v1"
    assert profile.acquisition_policy_id == "neural_v006"
    assert profile.play_policy_id == "heuristic_v008"
    assert profile.banish_policy_id == "deterministic_blaster_crystal"


def test_hybrid_player_runs_behind_legacy_middleware() -> None:
    from shards_ai.ai import build_hybrid_player

    game = _game()
    hybrid_id = game.active_player
    hybrid = build_hybrid_player(
        hybrid_id,
        game,
        GameRandom(8802),
        profile="hybrid-v001",
    )
    opponent = RandomPlayer(hybrid_id.opponent, GameRandom(8803))

    final_state = GameRunner(
        game,
        {hybrid_id: hybrid, hybrid_id.opponent: opponent},
        max_actions=20,
        max_turns=1,
    ).run()

    assert final_state.turn_number >= 1


def test_versioned_hybrid_profile_rejects_unknown_policy(tmp_path) -> None:
    profile_path = tmp_path / "invalid.yaml"
    profile_path.write_text(
        """
schema_version: 1
profile_id: invalid
policies:
  acquisition:
    policy_id: neural_v999
    checkpoint: configs/neural_profiles/v006.pt
  play:
    policy_id: heuristic_v008
    profile: configs/heuristic_profiles/v008.yaml
  banish:
    policy_id: deterministic_blaster_crystal
""",
        encoding="utf-8",
    )

    from shards_ai.ai import build_hybrid_player

    with pytest.raises(ValueError, match="Unsupported acquisition policy"):
        build_hybrid_player(
            PlayerId.PLAYER_1,
            _game(),
            GameRandom(9),
            profile=profile_path,
        )


def _play_card(card_id: str, effect: Effect, faction: Faction | None = None) -> CardInstance:
    return CardInstance(
        f"instance-{card_id}",
        CardDefinition(card_id, card_id, 0, effect, faction=faction),
    )


def test_algorithmic_play_prioritizes_safe_draw_before_other_cards() -> None:
    game = _game()
    game.state.phase = Phase.PLAY
    game.active.hand = [
        _play_card("plain", Effect(gems=1)),
        _play_card(
            "safe-draw",
            Effect(steps=(EffectStep((Operation("draw_card", amount=1),)),)),
        ),
        _play_card(
            "banish-offer",
            Effect(steps=(EffectStep((Operation("offer_banish", amount=1),)),)),
        ),
    ]
    game.active.draw_pile = [CardInstance("draw", CRYSTAL)]
    actions = game.legal_actions()

    action, reason = AlgorithmicPlayPolicy(PlayerId.PLAYER_1).choose_action(
        game.state, actions
    )

    assert action == PlayCard("instance-safe-draw")
    assert reason == "priority_1"


def test_algorithmic_play_prioritizes_spectra_echo_after_banish_offer() -> None:
    game = _game()
    game.state.phase = Phase.PLAY
    game.active.discard_pile = [CardInstance("old-spectra", _play_card("old", Effect(gems=1), Faction.SPECTRA).definition)]
    game.active.hand = [
        _play_card(
            "echo-spectra",
            Effect(steps=(EffectStep((Operation("draw_card", amount=1, requires_echo=True),)),)),
            Faction.SPECTRA,
        ),
        _play_card(
            "banish-offer",
            Effect(steps=(EffectStep((Operation("offer_banish", amount=1),)),)),
        ),
    ]
    game.active.draw_pile = []

    policy = AlgorithmicPlayPolicy(PlayerId.PLAYER_1)
    action, _ = policy.choose_action(game.state, game.legal_actions())

    assert action == PlayCard("instance-banish-offer")
    game.apply(action)
    # The engine exposes the pending banishment to the independent banish policy;
    # the next PLAY ranking is evaluated only after that decision is resolved.
    assert game.active.pending_banishes == 1
