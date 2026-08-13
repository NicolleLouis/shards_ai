"""Composable player with independently replaceable decision policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shards_ai.game import Game
from shards_ai.game.actions import (
    Action,
    BanishCard,
    BuyCard,
    RecruitFreeCard,
    RecruitMercenary,
    SkipBanish,
    StopBuying,
)
from shards_ai.game.enums import PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.state import GameState

from .heuristic_player import HeuristicPlayer
from .heuristic_profiles import HeuristicProfile, load_profile
from .hybrid_profiles import HybridProfile, load_hybrid_profile
from .macro_player import MacroNeuralPlayer
from .algorithmic_play import AlgorithmicPlayPolicy


ACQUISITION_ACTION_TYPES = (BuyCard, RecruitMercenary, RecruitFreeCard, StopBuying)


@dataclass(frozen=True, slots=True)
class DecisionDiagnostic:
    """Description of the policy that produced the last action."""

    policy_id: str
    decision_family: str
    action_type: str
    fallback_used: bool = False
    reason: str | None = None
    chosen_score: float | None = None
    ranked_alternatives: tuple[dict[str, object], ...] = ()


class DeterministicBanishPolicy:
    """Banish Blaster, then discard Crystal, otherwise skip.

    The policy only selects among actions supplied by the game engine. Card
    definitions are compared by stable card IDs and ties by instance ID.
    """

    policy_id = "deterministic_blaster_crystal"

    def __init__(self, policy_id: str | None = None) -> None:
        if policy_id is not None:
            self.policy_id = policy_id

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> tuple[Action, str]:
        actions = list(legal_actions)
        banishes = [action for action in actions if isinstance(action, BanishCard)]
        if not banishes:
            if any(isinstance(action, SkipBanish) for action in actions):
                return SkipBanish(), "no_preferred_banish_target"
            raise InvalidActionError("Banish policy received no legal banish action")

        player = observation.players[observation.active_player]
        def matching(card_id: str, *, discard_only: bool = False) -> list[BanishCard]:
            allowed_ids = {
                card.instance_id
                for card in (player.discard_pile if discard_only else (*player.hand, *player.discard_pile))
                if card.definition.card_id == card_id
            }
            return sorted(
                (action for action in banishes if action.card_id in allowed_ids),
                key=lambda action: action.card_id,
            )

        blasters = matching("blaster")
        if blasters:
            return blasters[0], "preferred_blaster"

        crystals = matching("crystal", discard_only=True)
        if crystals:
            return crystals[0], "preferred_discard_crystal"

        if any(isinstance(action, SkipBanish) for action in actions):
            return SkipBanish(), "no_preferred_banish_target"

        # The engine normally exposes SkipBanish, but fail loudly if that
        # contract changes instead of silently selecting another card.
        raise InvalidActionError(
            f"No preferred banish target and SkipBanish is unavailable: {actions!r}"
        )


class HeuristicPlayPolicy:
    """Fixed PLAY policy backed by the protected Heuristic V008 profile."""

    policy_id = "heuristic_v008"

    def __init__(self, player_id: PlayerId, profile: HeuristicProfile) -> None:
        self._player = HeuristicPlayer(
            player_id,
            weights=profile.weights,
            acquisition_weights=profile.card_acquisition_weights,
            constraint_weights=profile.constraint_weights,
        )

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> tuple[Action, str]:
        return self._player.choose_action(observation, legal_actions), "heuristic_v008"


class NeuralAcquisitionPolicy:
    """Use the V006 macro scorer for BUY and recruitment decisions only."""

    policy_id = "neural_v006"

    def __init__(self, player: MacroNeuralPlayer, game: Game) -> None:
        self._player = player
        self._game = game
        self.last_chosen_score: float | None = None
        self.last_ranked_alternatives: tuple[dict[str, object], ...] = ()

    @property
    def decisions(self) -> int:
        return self._player.decisions

    @property
    def total_inference_seconds(self) -> float:
        return self._player.total_inference_seconds

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> tuple[Action, str]:
        neural_observation = self._game.neural_observation_for(self._player.player_id)
        action = self._player.choose_action(neural_observation, legal_actions)
        candidates = self._player.last_scored_candidates
        scores = self._player.last_candidate_scores
        ranked = []
        for index, candidate in enumerate(candidates):
            if index >= len(scores) or not candidate.atomic_trace:
                continue
            ranked.append({
                "action_type": type(candidate.atomic_trace[0]).__name__,
                "action": repr(candidate.atomic_trace[0]),
                "score": round(scores[index], 6),
                "selected": candidate.atomic_trace[0] == action,
            })
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        self.last_ranked_alternatives = tuple(ranked)
        selected = next((item for item in ranked if item["selected"]), None)
        self.last_chosen_score = float(selected["score"]) if selected else None
        return action, "neural_v006"


class HybridPlayer:
    """Versioned composition of independent acquisition, PLAY and banish policies."""

    observation_kind = "game_state"
    observation_is_read_only = True

    def __init__(
        self,
        player_id: PlayerId,
        game: Game,
        *,
        acquisition_policy: NeuralAcquisitionPolicy,
        play_policy: HeuristicPlayPolicy | AlgorithmicPlayPolicy,
        banish_policy: DeterministicBanishPolicy | None = None,
    ) -> None:
        self.player_id = player_id
        self.game = game
        self.acquisition_policy = acquisition_policy
        self.play_policy = play_policy
        self.banish_policy = banish_policy or DeterministicBanishPolicy()
        self.last_decision: DecisionDiagnostic | None = None

    @property
    def decisions(self) -> int:
        return self.acquisition_policy.decisions

    @property
    def total_inference_seconds(self) -> float:
        return self.acquisition_policy.total_inference_seconds

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise InvalidActionError("Cannot choose an action from an empty action list")
        if not isinstance(observation, GameState):
            raise TypeError("ComposedPlayer requires a GameState")

        if any(isinstance(action, (BanishCard, SkipBanish)) for action in actions):
            policy = self.banish_policy
            family = "banish"
        elif any(isinstance(action, ACQUISITION_ACTION_TYPES) for action in actions):
            policy = self.acquisition_policy
            family = "acquisition"
        else:
            policy = self.play_policy
            family = "play"

        action, reason = policy.choose_action(observation, actions)
        if action not in actions:
            raise InvalidActionError(
                f"Policy {policy.policy_id} returned illegal action: {action!r}"
            )
        self.last_decision = DecisionDiagnostic(
            policy_id=policy.policy_id,
            decision_family=family,
            action_type=type(action).__name__,
            reason=reason,
            chosen_score=getattr(policy, "last_chosen_score", None),
            ranked_alternatives=getattr(policy, "last_ranked_alternatives", ()),
        )
        return action


def build_composed_player(
    player_id: PlayerId,
    game: Game,
    rng,
    *,
    acquisition_checkpoint: str | Path = "configs/neural_profiles/v006.pt",
    play_profile: str | Path = "configs/heuristic_profiles/v008.yaml",
    play_policy_id: str = "heuristic_v008",
    banish_policy_id: str = "deterministic_blaster_crystal",
) -> HybridPlayer:
    """Build the initial V006 acquisition / V008 play composition."""

    from .player_factory import build_neural_player

    acquisition_player = build_neural_player(
        player_id,
        game,
        rng,
        checkpoint_path=acquisition_checkpoint,
    )
    if not isinstance(acquisition_player, MacroNeuralPlayer):
        raise ValueError(
            "The acquisition checkpoint must use the V006 macro scorer contract"
        )
    if play_policy_id == "heuristic_v008":
        play_policy = HeuristicPlayPolicy(player_id, load_profile(play_profile))
    elif play_policy_id == "algorithmic_play_v001":
        play_policy = AlgorithmicPlayPolicy(player_id)
    else:
        raise ValueError(f"Unsupported play policy: {play_policy_id!r}")
    if banish_policy_id not in {
        "deterministic_blaster_crystal",
        "deterministic_blaster_crystal_v001",
    }:
        raise ValueError(f"Unsupported banish policy: {banish_policy_id!r}")
    return HybridPlayer(
        player_id,
        game,
        acquisition_policy=NeuralAcquisitionPolicy(acquisition_player, game),
        play_policy=play_policy,
        banish_policy=DeterministicBanishPolicy(banish_policy_id),
    )


def build_hybrid_player(
    player_id: PlayerId,
    game: Game,
    rng,
    *,
    profile: str | Path | HybridProfile = "hybrid-v001",
) -> HybridPlayer:
    """Build one exact, replayable hybrid profile."""

    selected = profile if isinstance(profile, HybridProfile) else load_hybrid_profile(profile)
    if selected.acquisition_policy_id != "neural_v006":
        raise ValueError(
            f"Unsupported acquisition policy for current HybridPlayer: "
            f"{selected.acquisition_policy_id!r}"
        )
    if selected.play_policy_id not in {"heuristic_v008", "algorithmic_play_v001"}:
        raise ValueError(f"Unsupported play policy for current HybridPlayer: {selected.play_policy_id!r}")
    if selected.banish_policy_id not in {
        "deterministic_blaster_crystal",
        "deterministic_blaster_crystal_v001",
    }:
        raise ValueError(
            f"Unsupported banish policy for current HybridPlayer: "
            f"{selected.banish_policy_id!r}"
        )
    return build_composed_player(
        player_id,
        game,
        rng,
        acquisition_checkpoint=selected.acquisition_checkpoint,
        play_profile=selected.play_profile,
        play_policy_id=selected.play_policy_id,
        banish_policy_id=selected.banish_policy_id,
    )


__all__ = [
    "HybridPlayer",
    "DecisionDiagnostic",
    "DeterministicBanishPolicy",
    "HeuristicPlayPolicy",
    "NeuralAcquisitionPolicy",
    "build_composed_player",
    "build_hybrid_player",
]


# Compatibility aliases for the initial implementation name. New code should
# use HybridPlayer and build_hybrid_player/profile-based construction.
ComposedPlayer = HybridPlayer
