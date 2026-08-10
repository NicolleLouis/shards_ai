"""Action-conditioned tactical features for the V005 deck-state scorer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from shards_ai.game.observation import NeuralObservation

from .action_representation import ActionRepresentation
from .structured_v005 import StructuredSemanticV5DeckStateScorer


_TACTICAL_FEATURE_SIZE = 7
_DOMINATION_FACTIONS = ("maquis", "spectra", "homodeus")


class StructuredSemanticV6TacticalActionScorer(StructuredSemanticV5DeckStateScorer):
    """V005 scorer with candidate-specific Union, Echo and Domination features."""

    architecture = "structured_semantic_v6_tactical_action_v1"

    def __init__(self, config=None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)

    def _action_feature_size(self) -> int:
        return _TACTICAL_FEATURE_SIZE

    def _action_features(
        self,
        observation: NeuralObservation | None,
        actions: Sequence[ActionRepresentation],
    ) -> Tensor:
        if observation is None:
            if self.training:
                raise ValueError("Tactical action encoding requires an observation")
            return torch.zeros(
                (len(actions), _TACTICAL_FEATURE_SIZE),
                dtype=self.card_id_embedding.weight.dtype,
                device=self.device,
            )
        return torch.tensor(
            [self._features_for_action(observation, action) for action in actions],
            dtype=self.card_id_embedding.weight.dtype,
            device=self.device,
        )

    def _features_for_action(
        self,
        observation: NeuralObservation,
        action: ActionRepresentation,
    ) -> tuple[float, ...]:
        if action.action_type != "play_card" or action.card_definition_id is None:
            return (0.0,) * _TACTICAL_FEATURE_SIZE
        representation = self._semantic_cache.get(action.card_definition_id)
        if representation is None:
            return (0.0,) * _TACTICAL_FEATURE_SIZE

        requirements = self._active_requirements(observation, representation)
        union_required, echo_required, domination_required = requirements
        union_active = union_required and self._union_active(observation, action)
        echo_active = echo_required and self._echo_active(observation)
        missing = self._domination_missing_count(observation, action)
        domination_active = domination_required and missing == 0
        return (
            float(union_required),
            float(union_active),
            float(echo_required),
            float(echo_active),
            float(domination_required),
            float(domination_active),
            missing / len(_DOMINATION_FACTIONS),
        )

    @staticmethod
    def _active_requirements(observation, representation) -> tuple[bool, bool, bool]:
        requirements = [False, False, False]
        effects = [representation.effect]
        if representation.on_play_effect is not None:
            effects.append(representation.on_play_effect)
        for effect in effects:
            active_step = next(
                (
                    step for step in effect.steps
                    if step.mastery_at_least is None
                    or observation.active_player.mastery >= step.mastery_at_least
                ),
                None,
            )
            if active_step is None:
                continue
            for operation in active_step.operations:
                if operation.mastery_at_least is not None and observation.active_player.mastery < operation.mastery_at_least:
                    continue
                if operation.health_at_least is not None and observation.active_player.health < operation.health_at_least:
                    continue
                requirements[0] |= operation.requires_union
                requirements[1] |= operation.requires_echo
                requirements[2] |= operation.requires_domination
        return tuple(requirements)

    @staticmethod
    def _union_active(observation: NeuralObservation, action: ActionRepresentation) -> bool:
        candidate_id = action.card_instance_id
        candidate = next(
            (card for card in observation.active_player.hand if card.instance_id == candidate_id),
            None,
        )
        if candidate is None or candidate.faction is None:
            return False
        return any(
            card.instance_id != candidate_id and card.faction == candidate.faction
            for card in (*observation.active_player.hand, *observation.active_player.play_zone)
        )

    @staticmethod
    def _echo_active(observation: NeuralObservation) -> bool:
        return any(card.faction == "spectra" for card in observation.active_player.discard)

    @staticmethod
    def _domination_missing_count(
        observation: NeuralObservation,
        action: ActionRepresentation,
    ) -> int:
        candidate_id = action.card_instance_id
        factions = {
            card.faction
            for card in (*observation.active_player.hand, *observation.active_player.play_zone)
            if card.instance_id != candidate_id and card.faction in _DOMINATION_FACTIONS
        }
        factions.update(
            faction
            for faction, present in zip(
                _DOMINATION_FACTIONS,
                observation.active_player.played_champion_faction_mask[:3],
            )
            if present
        )
        return len(set(_DOMINATION_FACTIONS) - factions)
