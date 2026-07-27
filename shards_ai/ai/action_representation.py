from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from shards_ai.game.actions import (
    Action,
    ActivateChampion,
    AssignPower,
    BanishCard,
    BuyCard,
    ChoosePendingDecision,
    GainMastery,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
    SkipBanish,
    StopBuying,
)
from shards_ai.game.cards import CardInstance
from shards_ai.game.state import GameState
from shards_ai.game.observation import NeuralCardObservation, NeuralObservation


ACTION_REPRESENTATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ActionRepresentation:
    action_type: str
    phase: str
    card_definition_id: str | None = None
    card_instance_id: str | None = None
    river_slot: int | None = None
    target: str | None = None
    amount: int | None = None
    choice_id: str | None = None
    schema_version: int = ACTION_REPRESENTATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def representation_for_action(action: Action, state: GameState) -> ActionRepresentation:
    """Build a semantic representation without mutating or applying ``action``."""

    phase = state.phase.value
    if isinstance(action, PlayCard):
        card = _find_card(state.players[state.active_player].hand, action.card_id)
        return _card_action("play_card", phase, card, card_instance_id=action.card_id)
    if isinstance(action, ActivateChampion):
        card = _find_card(state.players[state.active_player].champions, action.champion_id)
        return _card_action(
            "activate_champion",
            phase,
            card,
            card_instance_id=action.champion_id,
        )
    if isinstance(action, BanishCard):
        player = state.players[state.active_player]
        card = _find_card([*player.hand, *player.discard_pile], action.card_id)
        return _card_action("banish_card", phase, card, card_instance_id=action.card_id)
    if isinstance(action, SkipBanish):
        return ActionRepresentation(action_type="skip_banish", phase=phase)
    if isinstance(action, RecruitFreeCard):
        card = _river_card(state, action.river_slot, action.card_instance_id)
        return _card_action(
            "recruit_free_card",
            phase,
            card,
            card_instance_id=action.card_instance_id,
            river_slot=action.river_slot,
        )
    if isinstance(action, PassPlayPhase):
        return ActionRepresentation(action_type="pass_play_phase", phase=phase)
    if isinstance(action, GainMastery):
        return ActionRepresentation(action_type="gain_mastery", phase=phase)
    if isinstance(action, BuyCard):
        card = _river_card(state, action.river_slot, action.card_instance_id)
        return _card_action(
            "buy_card",
            phase,
            card,
            card_instance_id=action.card_instance_id,
            river_slot=action.river_slot,
        )
    if isinstance(action, RecruitMercenary):
        card = _river_card(state, action.river_slot, action.card_instance_id)
        return _card_action(
            "recruit_mercenary",
            phase,
            card,
            card_instance_id=action.card_instance_id,
            river_slot=action.river_slot,
        )
    if isinstance(action, StopBuying):
        return ActionRepresentation(action_type="stop_buying", phase=phase)
    if isinstance(action, AssignPower):
        card = None
        card_instance_id = None
        if action.target != "opponent":
            card = _find_card(
                state.players[state.active_player.opponent].champions,
                action.target,
            )
            card_instance_id = action.target
        return _card_action(
            "assign_power",
            phase,
            card,
            card_instance_id=card_instance_id,
            target=action.target,
            amount=action.amount,
        )
    if isinstance(action, ChoosePendingDecision):
        card = _find_public_choice(state, action.choice_id)
        return _card_action(
            "choose_pending_decision",
            phase,
            card,
            card_instance_id=action.choice_id if card is not None else None,
            choice_id=action.choice_id,
        )
    raise ValueError(f"Unsupported action for representation: {action!r}")


def representation_for_neural_action(
    action: Action,
    observation: NeuralObservation,
) -> ActionRepresentation:
    """Represent an action using only cards visible in a masked observation."""

    phase = observation.phase
    active = observation.active_player
    opponent = observation.opponent
    if isinstance(action, PlayCard):
        card = _find_observation_card(active.hand, action.card_id)
        return _neural_card_action("play_card", phase, card, card_instance_id=action.card_id)
    if isinstance(action, ActivateChampion):
        card = _find_observation_card(active.champions, action.champion_id)
        return _neural_card_action("activate_champion", phase, card, card_instance_id=action.champion_id)
    if isinstance(action, BanishCard):
        card = _find_observation_card((*active.hand, *active.discard), action.card_id)
        return _neural_card_action("banish_card", phase, card, card_instance_id=action.card_id)
    if isinstance(action, SkipBanish):
        return ActionRepresentation("skip_banish", phase)
    if isinstance(action, RecruitFreeCard):
        card = _neural_river_card(observation, action.river_slot, action.card_instance_id)
        return _neural_card_action("recruit_free_card", phase, card, card_instance_id=action.card_instance_id, river_slot=action.river_slot)
    if isinstance(action, PassPlayPhase):
        return ActionRepresentation("pass_play_phase", phase)
    if isinstance(action, GainMastery):
        return ActionRepresentation("gain_mastery", phase)
    if isinstance(action, BuyCard):
        card = _neural_river_card(observation, action.river_slot, action.card_instance_id)
        return _neural_card_action("buy_card", phase, card, card_instance_id=action.card_instance_id, river_slot=action.river_slot)
    if isinstance(action, RecruitMercenary):
        card = _neural_river_card(observation, action.river_slot, action.card_instance_id)
        return _neural_card_action("recruit_mercenary", phase, card, card_instance_id=action.card_instance_id, river_slot=action.river_slot)
    if isinstance(action, StopBuying):
        return ActionRepresentation("stop_buying", phase)
    if isinstance(action, AssignPower):
        card = None if action.target == "opponent" else _find_observation_card(opponent.champions, action.target)
        return _neural_card_action("assign_power", phase, card, card_instance_id=None if card is None else action.target, target=action.target, amount=action.amount)
    if isinstance(action, ChoosePendingDecision):
        public_cards = (*active.hand, *active.play_zone, *active.champions, *opponent.champions,
                        *(item.card for item in observation.river if item.card is not None))
        card = _find_observation_card(public_cards, action.choice_id)
        return _neural_card_action("choose_pending_decision", phase, card, card_instance_id=action.choice_id if card else None, choice_id=action.choice_id)
    raise ValueError(f"Unsupported action for neural representation: {action!r}")


def _card_action(
    action_type: str,
    phase: str,
    card: CardInstance | None,
    *,
    card_instance_id: str | None = None,
    river_slot: int | None = None,
    target: str | None = None,
    amount: int | None = None,
    choice_id: str | None = None,
) -> ActionRepresentation:
    if card is None and card_instance_id is not None:
        raise ValueError(
            f"Cannot resolve public card {card_instance_id!r} for action {action_type!r}"
        )
    return ActionRepresentation(
        action_type=action_type,
        phase=phase,
        card_definition_id=card.definition.card_id if card is not None else None,
        card_instance_id=card_instance_id,
        river_slot=river_slot,
        target=target,
        amount=amount,
        choice_id=choice_id,
    )


def _find_card(cards: list[CardInstance], instance_id: str) -> CardInstance | None:
    return next((card for card in cards if card.instance_id == instance_id), None)


def _find_observation_card(
    cards: Sequence[NeuralCardObservation],
    instance_id: str,
) -> NeuralCardObservation | None:
    return next((card for card in cards if card.instance_id == instance_id), None)


def _neural_river_card(observation: NeuralObservation, slot: int, instance_id: str) -> NeuralCardObservation:
    if not 0 <= slot < len(observation.river):
        raise ValueError(f"Invalid river slot for neural action representation: {slot}")
    river_card = observation.river[slot].card
    if river_card is None or river_card.instance_id != instance_id:
        raise ValueError(f"River slot {slot} does not contain {instance_id!r}")
    return river_card


def _neural_card_action(
    action_type: str,
    phase: str,
    card: NeuralCardObservation | None,
    *,
    card_instance_id: str | None = None,
    river_slot: int | None = None,
    target: str | None = None,
    amount: int | None = None,
    choice_id: str | None = None,
) -> ActionRepresentation:
    if card is None and card_instance_id is not None and action_type not in {"assign_power", "choose_pending_decision"}:
        raise ValueError(f"Cannot resolve public card {card_instance_id!r} for neural action {action_type!r}")
    return ActionRepresentation(
        action_type=action_type,
        phase=phase,
        card_definition_id=card.card_definition_id if card is not None else None,
        card_instance_id=card_instance_id,
        river_slot=river_slot,
        target=target,
        amount=amount,
        choice_id=choice_id,
    )


def _river_card(state: GameState, slot: int, instance_id: str) -> CardInstance:
    if not 0 <= slot < len(state.river):
        raise ValueError(f"Invalid river slot for action representation: {slot}")
    card = state.river[slot]
    if card is None or card.instance_id != instance_id:
        raise ValueError(f"River slot {slot} does not contain {instance_id!r}")
    return card


def _find_public_choice(state: GameState, choice_id: str) -> CardInstance | None:
    active = state.players[state.active_player]
    opponent = state.players[state.active_player.opponent]
    public_cards = [
        *active.hand,
        *active.discard_pile,
        *active.play_zone,
        *active.champions,
        *opponent.champions,
        *[card for card in state.river if card is not None],
    ]
    return _find_card(public_cards, choice_id)
