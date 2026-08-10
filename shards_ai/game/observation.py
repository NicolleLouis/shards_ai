from __future__ import annotations

from dataclasses import dataclass

from .cards import CardInstance
from .enums import Faction, GameStatus, Phase, PlayerId
from .state import GameState, PendingDecision, PlayerState


CardCounts = tuple[tuple[str, int], ...]
OBSERVATION_SCHEMA_VERSION = 3
PLAYABLE_FACTIONS: tuple[Faction, ...] = (
    Faction.MAQUIS,
    Faction.SPECTRA,
    Faction.HOMODEUS,
    Faction.ORDER,
)


@dataclass(frozen=True, slots=True)
class NeuralCardObservation:
    """Public information about one individually visible card."""

    card_definition_id: str
    instance_id: str
    faction: str | None
    cost: int
    shield: int
    is_champion: bool
    champion_health: int | None
    is_mercenary: bool
    activated: bool = False


@dataclass(frozen=True, slots=True)
class NeuralRiverCardObservation:
    slot: int
    card: NeuralCardObservation | None


@dataclass(frozen=True, slots=True)
class NeuralPendingObservation:
    kind: str
    candidates: tuple[str, ...] = ()
    pending_banishes: int = 0
    pending_free_recruit_cost: int | None = None
    pending_free_recruit_to_hand: bool = False


@dataclass(frozen=True, slots=True)
class NeuralActivePlayerObservation:
    health: int
    mastery: int
    gems: int
    power: int
    hand: tuple[NeuralCardObservation, ...]
    draw_pile_counts: CardCounts
    discard_counts: CardCounts
    play_zone: tuple[NeuralCardObservation, ...]
    champions: tuple[NeuralCardObservation, ...]
    owned_card_counts: CardCounts
    played_faction_mask: tuple[bool, bool, bool, bool]
    played_champion_faction_mask: tuple[bool, bool, bool, bool] = (False, False, False, False)
    # Public active-player discard cards, sorted for deterministic serialization.
    # ``discard_counts`` remains the compact aggregate consumed by the model.
    discard: tuple[NeuralCardObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class NeuralOpponentObservation:
    health: int
    mastery: int
    owned_card_counts: CardCounts
    discard_counts: CardCounts
    champions: tuple[NeuralCardObservation, ...]


@dataclass(frozen=True, slots=True)
class NeuralObservation:
    """Information available to the active player at one decision point."""

    phase: str
    status: str
    winner: str | None
    turn_number: int
    active_player: NeuralActivePlayerObservation
    opponent: NeuralOpponentObservation
    central_deck_counts: CardCounts
    river: tuple[NeuralRiverCardObservation, ...]
    pending_decision: NeuralPendingObservation | None
    schema_version: int = OBSERVATION_SCHEMA_VERSION


def build_neural_observation(state: GameState) -> NeuralObservation:
    """Build a detached, information-masked observation from a game state."""

    active_player = state.players[state.active_player]
    opponent_player = state.players[state.active_player.opponent]
    return NeuralObservation(
        phase=state.phase.value,
        status=state.status.value,
        winner=_winner_from_active_view(state),
        turn_number=state.turn_number,
        active_player=_active_player_observation(active_player),
        opponent=_opponent_observation(opponent_player),
        central_deck_counts=_card_counts(state.central_deck),
        river=tuple(
            NeuralRiverCardObservation(
                slot=slot,
                card=_card_observation(card) if card is not None else None,
            )
            for slot, card in enumerate(state.river)
        ),
        pending_decision=_pending_observation(active_player, state),
    )


def _active_player_observation(player: PlayerState) -> NeuralActivePlayerObservation:
    return NeuralActivePlayerObservation(
        health=player.health,
        mastery=player.mastery,
        gems=player.gems,
        power=player.power,
        hand=tuple(_card_observation(card) for card in player.hand),
        draw_pile_counts=_card_counts(player.draw_pile),
        discard_counts=_card_counts(player.discard_pile),
        play_zone=tuple(_card_observation(card) for card in player.play_zone),
        champions=tuple(
            _card_observation(
                card,
                activated=card.instance_id in player.activated_champion_ids,
            )
            for card in player.champions
        ),
        owned_card_counts=_card_counts(_owned_cards(player)),
        played_faction_mask=_played_faction_mask(player),
        played_champion_faction_mask=_played_champion_faction_mask(player),
        discard=tuple(
            sorted(
                (_card_observation(card) for card in player.discard_pile),
                key=lambda card: (card.card_definition_id, card.instance_id),
            )
        ),
    )


def _opponent_observation(player: PlayerState) -> NeuralOpponentObservation:
    return NeuralOpponentObservation(
        health=player.health,
        mastery=player.mastery,
        owned_card_counts=_card_counts(_owned_cards(player)),
        discard_counts=_card_counts(player.discard_pile),
        champions=tuple(
            _card_observation(
                card,
                activated=card.instance_id in player.activated_champion_ids,
            )
            for card in player.champions
        ),
    )


def _card_observation(card: CardInstance, *, activated: bool = False) -> NeuralCardObservation:
    definition = card.definition
    return NeuralCardObservation(
        card_definition_id=definition.card_id,
        instance_id=card.instance_id,
        faction=definition.faction.value if definition.faction is not None else None,
        cost=definition.cost,
        shield=definition.shield,
        is_champion=definition.is_champion,
        champion_health=definition.champion_health,
        is_mercenary=definition.is_mercenary,
        activated=activated,
    )


def _owned_cards(player: PlayerState) -> list[CardInstance]:
    return [
        *player.hand,
        *player.draw_pile,
        *player.discard_pile,
        *player.play_zone,
        *player.champions,
    ]


def _card_counts(cards: list[CardInstance]) -> CardCounts:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.definition.card_id] = counts.get(card.definition.card_id, 0) + 1
    return tuple(sorted(counts.items()))


def _played_faction_mask(player: PlayerState) -> tuple[bool, bool, bool, bool]:
    cards_by_instance_id = {
        card.instance_id: card
        for card in _owned_cards(player)
    }
    played_factions = {
        cards_by_instance_id[card_id].definition.faction
        for card_id in player.played_card_ids_this_turn
        if card_id in cards_by_instance_id
    }
    return tuple(faction in played_factions for faction in PLAYABLE_FACTIONS)  # type: ignore[return-value]


def _played_champion_faction_mask(player: PlayerState) -> tuple[bool, bool, bool, bool]:
    played_champion_factions = {
        card.definition.faction
        for card in player.champions
        if card.instance_id in player.played_card_ids_this_turn
    }
    return tuple(
        faction in played_champion_factions for faction in PLAYABLE_FACTIONS
    )  # type: ignore[return-value]


def _pending_observation(
    player: PlayerState,
    state: GameState,
) -> NeuralPendingObservation | None:
    pending: PendingDecision | None = player.pending_decision
    if pending is not None:
        return NeuralPendingObservation(
            kind=pending.kind,
            candidates=pending.candidates,
        )
    if player.pending_free_recruit_cost is not None:
        return NeuralPendingObservation(
            kind="recruit_free_card",
            candidates=tuple(
                card.instance_id
                for card in state.river
                if card is not None
                and card.definition.cost <= player.pending_free_recruit_cost
            ),
            pending_free_recruit_cost=player.pending_free_recruit_cost,
            pending_free_recruit_to_hand=player.pending_free_recruit_to_hand,
        )
    if player.pending_banishes:
        return NeuralPendingObservation(
            kind="banish",
            candidates=tuple(card.instance_id for card in [*player.hand, *player.discard_pile]),
            pending_banishes=player.pending_banishes,
        )
    return None


def _winner_from_active_view(state: GameState) -> str | None:
    if state.winner is None:
        return None
    if state.winner is state.active_player:
        return "active"
    if state.winner is state.active_player.opponent:
        return "opponent"
    raise ValueError(f"Winner {state.winner!r} is not a player in the game")
