"""Compact action-conditioned PyTorch model for heuristic imitation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from shards_ai.game.cards import CARD_CATALOG
from shards_ai.game.observation import NeuralCardObservation, NeuralObservation

from .action_representation import ActionRepresentation
from .card_representation import (
    CardSemanticRepresentation,
    SUPPORTED_CHAMPION_ABILITY_KINDS,
    SUPPORTED_OPERATION_KINDS,
    representation_for_definition,
)


ACTION_TYPES = (
    "play_card", "activate_champion", "banish_card", "skip_banish",
    "recruit_free_card", "pass_play_phase", "gain_mastery", "buy_card",
    "recruit_mercenary", "stop_buying", "assign_power", "choose_pending_decision",
)
TARGET_TYPES = ("", "opponent", "champion", "card", "other")
PHASE_TYPES = ("", "play", "buy", "combat", "pending", "end")
CARD_ID_UNK = "<UNK>"
OBSERVATION_FEATURE_SETS = ("baseline", "zone_cardinality_v1", "deck_state_v1")
_ZONE_CARDINALITY_BOUNDS = (100, 100, 20, 100, 100, 100, 20)
_DECK_STATE_FACTIONS = ("maquis", "spectra", "homodeus", "order")
_DECK_STATE_FACTION_BOUND = 100
_STATE_CONTEXT_SIZE = 2
_TACTICAL_ACTION_FEATURE_SIZE = 7


@dataclass(frozen=True, slots=True)
class NeuralModelConfig:
    """Dimensions and vocabularies for the first baseline model."""

    card_embedding_dim: int = 32
    card_id_embedding_dim: int = 12
    semantic_hidden_dim: int = 48
    state_hidden_dim: int = 96
    action_hidden_dim: int = 48
    scorer_hidden_dim: int = 96
    candidate_context_dim: int = 32
    semantic_token_hidden_dim: int = 64
    semantic_attention_heads: int = 4
    card_fusion_id_scale: float = 1.0
    card_fusion_semantic_scale: float = 1.0
    card_fusion_normalization: str = "l2"
    card_fusion_normalization_epsilon: float = 1e-8
    observation_feature_set: str = "baseline"


def _card_feature_size() -> int:
    return 12 + 4 + len(SUPPORTED_OPERATION_KINDS) + len(SUPPORTED_CHAMPION_ABILITY_KINDS)


class NeuralActionScorer(nn.Module):
    """Score each legal action from the player's information-masked observation.

    The model is intentionally action-conditioned: it does not produce a fixed
    action vocabulary, so adding a legal action or a card does not require an
    output-layer rewrite.
    """

    architecture = "independent_action"

    def __init__(
        self,
        config: NeuralModelConfig | None = None,
        *,
        card_catalog: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NeuralModelConfig()
        if self.config.observation_feature_set not in OBSERVATION_FEATURE_SETS:
            raise ValueError(
                "Unsupported observation_feature_set: "
                f"{self.config.observation_feature_set!r}"
            )
        self.card_catalog = card_catalog or CARD_CATALOG
        self.card_ids = tuple(sorted(self.card_catalog))
        self.card_to_index = {card_id: index for index, card_id in enumerate(self.card_ids)}
        self.unk_card_index = len(self.card_ids)

        self.card_id_embedding = nn.Embedding(len(self.card_ids) + 1, self.config.card_id_embedding_dim)
        self.card_semantic_encoder = nn.Sequential(
            nn.Linear(_card_feature_size(), self.config.semantic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.semantic_hidden_dim, self.config.card_embedding_dim),
        )
        self.card_fusion = nn.Sequential(
            nn.Linear(self.config.card_id_embedding_dim + self.config.card_embedding_dim, self.config.card_embedding_dim),
            nn.ReLU(),
        )
        self.action_type_embedding = nn.Embedding(len(ACTION_TYPES), self.config.action_hidden_dim)
        self.phase_embedding = nn.Embedding(len(PHASE_TYPES), 12)
        self.target_embedding = nn.Embedding(len(TARGET_TYPES), 8)
        scalar_action_size = 5
        self.action_encoder = nn.Sequential(
            nn.Linear(self.config.card_embedding_dim + self.config.action_hidden_dim + 12 + 8 + scalar_action_size
                      + self._action_feature_size(),
                      self.config.action_hidden_dim),
            nn.ReLU(),
        )
        # Eleven pooled card groups: active visible/count zones, opponent public
        # aggregates/champions, central deck, and river. Scalars carry resources
        # and masks; the river slot remains available through the action encoder.
        state_input_size = (
            self.config.card_embedding_dim * 11
            + self._base_state_scalar_size()
            + self._observation_feature_size()
            + 2
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_size, self.config.state_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.state_hidden_dim, self.config.state_hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.config.state_hidden_dim + self.config.action_hidden_dim, self.config.scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.scorer_hidden_dim, 1),
        )
        self._semantic_cache = {
            card_id: representation_for_definition(definition)
            for card_id, definition in self.card_catalog.items()
        }
        self.register_buffer(
            "_semantic_feature_tensor",
            torch.tensor(
                [
                    _semantic_features(self._semantic_cache[card_id])
                    for card_id in self.card_ids
                ] + [[0.0] * _card_feature_size()],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        # Card embeddings depend only on the frozen scorer weights and the card
        # definition.  NeuralPlayer runs the scorer in eval/inference mode, so
        # retain the vectors between decisions instead of rebuilding them for
        # every observation.  Training keeps the uncached path because its
        # parameters change after every optimizer step.
        self._inference_card_embedding_cache: dict[str | None, Tensor] = {}
        # Action encodings depend only on the immutable representation and the
        # frozen scorer weights.  Reuse the batched result for representations
        # encountered again during evaluation; training keeps the original path.
        self._inference_action_encoding_cache: dict[ActionRepresentation, Tensor] = {}

    def forward(self, observation: NeuralObservation, actions: Sequence[ActionRepresentation]) -> Tensor:
        return self.scores_for_actions(observation, actions)

    def scores_for_actions(self, observation: NeuralObservation, actions: Sequence[ActionRepresentation]) -> Tensor:
        if not actions:
            return torch.empty(0, device=self.device)
        card_ids = list(self._observation_card_ids(observation))
        card_ids.extend(action.card_definition_id for action in actions)
        embedding_lookup = self._embedding_lookup(card_ids)
        state = self.encode_observation(observation, embedding_lookup=embedding_lookup)
        action = self.encode_actions(actions, observation=observation, embedding_lookup=embedding_lookup)
        state_batch = state.expand(len(actions), -1)
        return self.scorer(torch.cat((state_batch, action), dim=1)).squeeze(1)

    def encode_observation(
        self,
        observation: NeuralObservation,
        *,
        embedding_lookup: dict[str | None, Tensor] | None = None,
    ) -> Tensor:
        embedding_lookup = embedding_lookup or self._embedding_lookup(self._observation_card_ids(observation))
        active = observation.active_player
        opponent = observation.opponent
        pools = (
            self._pool_cards(active.hand, embedding_lookup),
            self._pool_counts(active.draw_pile_counts, embedding_lookup),
            self._pool_counts(active.discard_counts, embedding_lookup),
            self._pool_cards(active.play_zone, embedding_lookup),
            self._pool_cards(active.champions, embedding_lookup),
            self._pool_counts(active.owned_card_counts, embedding_lookup),
            self._pool_counts(opponent.owned_card_counts, embedding_lookup),
            self._pool_counts(opponent.discard_counts, embedding_lookup),
            self._pool_cards(opponent.champions, embedding_lookup),
            self._pool_counts(observation.central_deck_counts, embedding_lookup),
            self._pool_cards((item.card for item in observation.river if item.card is not None), embedding_lookup),
        )
        scalars = self._state_scalars(observation)
        context = torch.tensor(
            [self._index(PHASE_TYPES, observation.phase) / max(1, len(PHASE_TYPES) - 1),
             observation.turn_number / 1000], dtype=torch.float32, device=self.device,
        )
        return self.state_encoder(torch.cat((*pools, scalars, context)).unsqueeze(0))

    @staticmethod
    def _base_state_scalar_size() -> int:
        return 12

    def _observation_feature_size(self) -> int:
        if self.config.observation_feature_set == "zone_cardinality_v1":
            return len(_ZONE_CARDINALITY_BOUNDS)
        if self.config.observation_feature_set == "deck_state_v1":
            return len(_ZONE_CARDINALITY_BOUNDS) + len(_DECK_STATE_FACTIONS)
        return 0

    def _state_scalars(self, observation: NeuralObservation) -> Tensor:
        active = observation.active_player
        opponent = observation.opponent
        values = [
            active.health / 100,
            active.mastery / 100,
            active.gems / 10,
            active.power / 20,
            opponent.health / 100,
            opponent.mastery / 100,
            *[float(value) for value in active.played_faction_mask],
            len(opponent.champions) / 10,
            len(active.champions) / 10,
        ]
        if self.config.observation_feature_set in {"zone_cardinality_v1", "deck_state_v1"}:
            values.extend(self._zone_cardinality_values(observation))
        if self.config.observation_feature_set == "deck_state_v1":
            values.extend(self._faction_count_values(active.owned_card_counts))
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _action_feature_size(self) -> int:
        return 0

    @staticmethod
    def _zone_cardinality_values(observation: NeuralObservation) -> tuple[float, ...]:
        active = observation.active_player
        opponent = observation.opponent
        counts = (
            sum(count for _card_id, count in active.draw_pile_counts),
            sum(count for _card_id, count in active.discard_counts),
            len(active.champions),
            sum(count for _card_id, count in active.owned_card_counts),
            sum(count for _card_id, count in opponent.owned_card_counts),
            sum(count for _card_id, count in opponent.discard_counts),
            len(opponent.champions),
        )
        return tuple(
            min(max(count, 0), bound) / bound
            for count, bound in zip(counts, _ZONE_CARDINALITY_BOUNDS)
        )

    def _faction_count_values(self, counts: Iterable[tuple[str, int]]) -> tuple[float, ...]:
        faction_counts = dict.fromkeys(_DECK_STATE_FACTIONS, 0)
        for card_id, count in counts:
            definition = self.card_catalog.get(card_id)
            faction = getattr(getattr(definition, "faction", None), "value", None)
            if faction in faction_counts:
                faction_counts[faction] += count
        return tuple(
            min(max(faction_counts[faction], 0), _DECK_STATE_FACTION_BOUND)
            / _DECK_STATE_FACTION_BOUND
            for faction in _DECK_STATE_FACTIONS
        )

    def encode_actions(
        self,
        actions: Sequence[ActionRepresentation],
        *,
        observation: NeuralObservation | None = None,
        embedding_lookup: dict[str | None, Tensor] | None = None,
    ) -> Tensor:
        has_context = observation is not None and self._action_feature_size() > 0
        if not has_context and not self.training and not torch.is_grad_enabled():
            unique_actions = list(dict.fromkeys(actions))
            missing_actions = [
                action for action in unique_actions
                if action not in self._inference_action_encoding_cache
            ]
            if missing_actions:
                computed = self._encode_actions_uncached(
                    missing_actions,
                    observation=observation,
                    embedding_lookup=embedding_lookup,
                )
                self._inference_action_encoding_cache.update(
                    zip(missing_actions, computed.unbind())
                )
            return torch.stack(
                [self._inference_action_encoding_cache[action] for action in actions]
            )
        return self._encode_actions_uncached(
            actions, observation=observation, embedding_lookup=embedding_lookup
        )

    def _encode_actions_uncached(
        self,
        actions: Sequence[ActionRepresentation],
        *,
        observation: NeuralObservation | None = None,
        embedding_lookup: dict[str | None, Tensor] | None = None,
    ) -> Tensor:
        embedding_lookup = embedding_lookup or self._embedding_lookup(action.card_definition_id for action in actions)
        card = torch.stack([embedding_lookup[action.card_definition_id] for action in actions])
        action_type = self.action_type_embedding(torch.tensor(
            [self._index(ACTION_TYPES, action.action_type) for action in actions],
            dtype=torch.long, device=self.device,
        ))
        phase = self.phase_embedding(torch.tensor(
            [self._index(PHASE_TYPES, action.phase) for action in actions],
            dtype=torch.long, device=self.device,
        ))
        target = self.target_embedding(torch.tensor(
            [self._index(TARGET_TYPES, self._target_type(action.target)) for action in actions],
            dtype=torch.long, device=self.device,
        ))
        scalars = torch.tensor([
            [float(action.amount or 0) / 20, float(action.river_slot or -1) / 6,
             float(action.card_definition_id is not None), float(action.choice_id is not None),
             float(action.card_instance_id is not None)]
            for action in actions
        ], dtype=torch.float32, device=self.device)
        tactical = self._action_features(observation, actions)
        return self.action_encoder(torch.cat((card, action_type, phase, target, scalars, tactical), dim=1))

    def _action_features(
        self,
        observation: NeuralObservation | None,
        actions: Sequence[ActionRepresentation],
    ) -> Tensor:
        return torch.zeros(
            (len(actions), self._action_feature_size()),
            dtype=self.card_id_embedding.weight.dtype,
            device=self.device,
        )

    @property
    def device(self) -> torch.device:
        return self.card_id_embedding.weight.device

    def _card_embedding(self, card_id: str | None) -> Tensor:
        return self._card_embeddings([card_id])[0]

    def _embedding_lookup(self, card_ids: Iterable[str | None]) -> dict[str | None, Tensor]:
        unique_ids = list(dict.fromkeys(card_ids))
        return dict(zip(unique_ids, self._card_embeddings(unique_ids)))

    def _card_embeddings(self, card_ids: Sequence[str | None]) -> Tensor:
        if not self.training and not torch.is_grad_enabled():
            missing_ids = [
                card_id for card_id in dict.fromkeys(card_ids)
                if card_id not in self._inference_card_embedding_cache
            ]
            if missing_ids:
                computed = self._card_embeddings_uncached(missing_ids)
                self._inference_card_embedding_cache.update(
                    zip(missing_ids, computed.unbind())
                )
            return torch.stack(
                [self._inference_card_embedding_cache[card_id] for card_id in card_ids]
            )
        return self._card_embeddings_uncached(card_ids)

    def _card_embeddings_uncached(self, card_ids: Sequence[str | None]) -> Tensor:
        indices = torch.tensor(
            [self.card_to_index.get(card_id or "", self.unk_card_index) for card_id in card_ids],
            dtype=torch.long, device=self.device,
        )
        semantic_features = self._semantic_feature_tensor.index_select(0, indices)
        id_vectors = self.card_id_embedding(indices)
        semantic_vectors = self.card_semantic_encoder(semantic_features)
        return self.card_fusion(torch.cat((id_vectors, semantic_vectors), dim=1))

    def _pool_cards(
        self,
        cards: Iterable[NeuralCardObservation],
        embedding_lookup: dict[str | None, Tensor],
    ) -> Tensor:
        card_ids = [card.card_definition_id for card in cards]
        return self._pool_card_ids(card_ids, embedding_lookup)

    def _pool_counts(
        self,
        counts: Iterable[tuple[str, int]],
        embedding_lookup: dict[str | None, Tensor],
    ) -> Tensor:
        card_ids = []
        for card_id, count in counts:
            card_ids.extend([card_id] * min(count, 20))
        return self._pool_card_ids(card_ids, embedding_lookup)

    def _pool_card_ids(
        self,
        card_ids: Sequence[str | None],
        embedding_lookup: dict[str | None, Tensor],
    ) -> Tensor:
        if not card_ids:
            return torch.zeros(self.config.card_embedding_dim, device=self.device)
        return torch.stack([embedding_lookup[card_id] for card_id in card_ids]).mean(dim=0)

    @staticmethod
    def _observation_card_ids(observation: NeuralObservation) -> Iterable[str | None]:
        active = observation.active_player
        opponent = observation.opponent
        yield from (card.card_definition_id for card in active.hand)
        yield from (card_id for card_id, _count in active.draw_pile_counts)
        yield from (card_id for card_id, _count in active.discard_counts)
        yield from (card.card_definition_id for card in active.play_zone)
        yield from (card.card_definition_id for card in active.champions)
        yield from (card_id for card_id, _count in active.owned_card_counts)
        yield from (card_id for card_id, _count in opponent.owned_card_counts)
        yield from (card_id for card_id, _count in opponent.discard_counts)
        yield from (card.card_definition_id for card in opponent.champions)
        yield from (card_id for card_id, _count in observation.central_deck_counts)
        yield from (item.card.card_definition_id for item in observation.river if item.card is not None)

    @staticmethod
    def _index(values: Sequence[str], value: str) -> int:
        return values.index(value) if value in values else 0

    @staticmethod
    def _target_type(target: str | None) -> str:
        if target in TARGET_TYPES:
            return target or ""
        return "champion" if target else ""


class ContextualNeuralActionScorer(NeuralActionScorer):
    """Score actions after pooling the alternatives available in the decision.

    The pooling is permutation-invariant and intentionally lighter than a Transformer: every
    action sees the same summary of the candidate set, but candidates do not attend to one another.
    """

    architecture = "global_candidate_context"

    def __init__(
        self,
        config: NeuralModelConfig | None = None,
        *,
        card_catalog: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(config, card_catalog=card_catalog)
        self.candidate_context_encoder = nn.Sequential(
            nn.Linear(self.config.action_hidden_dim, self.config.candidate_context_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(
                self.config.state_hidden_dim
                + self.config.action_hidden_dim
                + self.config.candidate_context_dim,
                self.config.scorer_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.config.scorer_hidden_dim, 1),
        )

    def scores_for_actions(
        self,
        observation: NeuralObservation,
        actions: Sequence[ActionRepresentation],
    ) -> Tensor:
        if not actions:
            return torch.empty(0, device=self.device)
        card_ids = list(self._observation_card_ids(observation))
        card_ids.extend(action.card_definition_id for action in actions)
        embedding_lookup = self._embedding_lookup(card_ids)
        state = self.encode_observation(observation, embedding_lookup=embedding_lookup)
        action = self.encode_actions(actions, observation=observation, embedding_lookup=embedding_lookup)
        context = self.candidate_context_encoder(action.mean(dim=0, keepdim=True))
        state_batch = state.expand(len(actions), -1)
        context_batch = context.expand(len(actions), -1)
        return self.scorer(torch.cat((state_batch, action, context_batch), dim=1)).squeeze(1)


class SemanticIdentityNeuralActionScorer(NeuralActionScorer):
    """V003 scorer with a wider, explicitly versioned card representation."""

    architecture = "semantic_identity_v3"


def migrate_v004_checkpoint_to_deck_state(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Expand a V004 checkpoint with the combined deck-state features."""
    source_config = NeuralModelConfig(**checkpoint["model_config"])
    if source_config.observation_feature_set != "baseline":
        raise ValueError("Migration source must use the baseline observation feature set")
    source_architecture = str(checkpoint.get("architecture", "independent_action"))
    if source_architecture != "structured_semantic_v5_fusion_experiment":
        raise ValueError("Migration source must be the V004 structured semantic architecture")

    target_config = replace(source_config, observation_feature_set="deck_state_v1")
    extra_features = len(_ZONE_CARDINALITY_BOUNDS) + len(_DECK_STATE_FACTIONS)
    migrated = dict(checkpoint)
    migrated["model_config"] = asdict(target_config)
    migrated["architecture"] = "structured_semantic_v5_deck_state_v1"
    migrated["migration"] = {
        "source_architecture": source_architecture,
        "source_observation_feature_set": source_config.observation_feature_set,
        "target_observation_feature_set": target_config.observation_feature_set,
        "zero_initialized_state_features": extra_features,
    }

    for state_key in ("model_state_dict", "actor_critic_state_dict"):
        state = checkpoint.get(state_key)
        if not isinstance(state, Mapping):
            continue
        expanded_state = dict(state)
        for key, value in state.items():
            if not key.endswith("state_encoder.0.weight"):
                continue
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"Unexpected state encoder weight for migration: {key}")
            prefix_size = value.shape[1] - _STATE_CONTEXT_SIZE
            if prefix_size <= 0:
                raise ValueError(f"Invalid state encoder input size for migration: {key}")
            expanded = value.new_zeros(value.shape[0], value.shape[1] + extra_features)
            expanded[:, :prefix_size] = value[:, :prefix_size]
            expanded[:, prefix_size + extra_features:] = value[:, prefix_size:]
            expanded_state[key] = expanded
        migrated[state_key] = expanded_state
    return migrated


def migrate_v005_deck_state_checkpoint_to_tactical(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Expand a V005 deck-state checkpoint with tactical action features."""
    source_config = NeuralModelConfig(**checkpoint["model_config"])
    if source_config.observation_feature_set != "deck_state_v1":
        raise ValueError("Migration source must use the deck_state_v1 observation feature set")
    source_architecture = str(checkpoint.get("architecture", "independent_action"))
    if source_architecture != "structured_semantic_v5_deck_state_v1":
        raise ValueError("Migration source must be the V005 deck-state architecture")

    migrated = dict(checkpoint)
    migrated["architecture"] = "structured_semantic_v6_tactical_action_v1"
    migrated["migration"] = {
        "source_architecture": source_architecture,
        "target_architecture": migrated["architecture"],
        "zero_initialized_action_features": _TACTICAL_ACTION_FEATURE_SIZE,
    }
    for state_key in ("model_state_dict", "actor_critic_state_dict"):
        state = checkpoint.get(state_key)
        if not isinstance(state, Mapping):
            continue
        expanded_state = dict(state)
        for key, value in state.items():
            if not key.endswith("action_encoder.0.weight"):
                continue
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"Unexpected action encoder weight for migration: {key}")
            expanded = value.new_zeros(
                value.shape[0], value.shape[1] + _TACTICAL_ACTION_FEATURE_SIZE
            )
            expanded[:, :value.shape[1]] = value
            expanded_state[key] = expanded
        migrated[state_key] = expanded_state
    return migrated


SUPPORTED_ARCHITECTURES = (
    NeuralActionScorer.architecture,
    ContextualNeuralActionScorer.architecture,
    SemanticIdentityNeuralActionScorer.architecture,
    "structured_semantic_v4",
    "structured_semantic_v5_fusion_experiment",
    "structured_semantic_v5_deck_state_v1",
    "structured_semantic_v6_tactical_action_v1",
)


def build_neural_scorer(
    architecture: str,
    config: NeuralModelConfig | None = None,
    *,
    card_catalog: Mapping[str, object] | None = None,
) -> NeuralActionScorer:
    """Build the scorer selected by explicit checkpoint/profile metadata."""

    classes = {
        NeuralActionScorer.architecture: NeuralActionScorer,
        ContextualNeuralActionScorer.architecture: ContextualNeuralActionScorer,
        SemanticIdentityNeuralActionScorer.architecture: SemanticIdentityNeuralActionScorer,
    }
    if architecture == "structured_semantic_v4":
        from .structured_v004 import StructuredSemanticV4Scorer

        return StructuredSemanticV4Scorer(config, card_catalog=card_catalog)
    if architecture == "structured_semantic_v5_fusion_experiment":
        from .structured_v005 import StructuredSemanticV5FusionScorer

        return StructuredSemanticV5FusionScorer(config, card_catalog=card_catalog)
    if architecture == "structured_semantic_v5_deck_state_v1":
        from .structured_v005 import StructuredSemanticV5DeckStateScorer

        return StructuredSemanticV5DeckStateScorer(config, card_catalog=card_catalog)
    if architecture == "structured_semantic_v6_tactical_action_v1":
        from .structured_v006 import StructuredSemanticV6TacticalActionScorer

        return StructuredSemanticV6TacticalActionScorer(config, card_catalog=card_catalog)
    try:
        model_class = classes[architecture]
    except KeyError as error:
        raise ValueError(f"Unsupported neural architecture: {architecture!r}") from error
    return model_class(config, card_catalog=card_catalog)


def _semantic_features(card: CardSemanticRepresentation) -> list[float]:
    operations = [operation.kind for step in card.effect.steps for operation in step.operations]
    if card.on_play_effect is not None:
        operations.extend(operation.kind for step in card.on_play_effect.steps for operation in step.operations)
    abilities = [card.champion_ability.kind] if card.champion_ability is not None else []
    factions = ("maquis", "spectra", "homodeus", "order")
    return [
        card.cost / 20, card.shield / 20, float(card.is_champion), (card.champion_health or 0) / 100,
        float(card.is_mercenary), float(card.effect.flat_gems), float(card.effect.flat_power),
        float(card.on_play_effect is not None), float(card.champion_ability is not None),
        float(card.passive_kind is not None), len(operations) / 10, len(abilities),
        *[float(card.faction == faction) for faction in factions],
        *[operations.count(kind) / 5 for kind in sorted(SUPPORTED_OPERATION_KINDS)],
        *[abilities.count(kind) for kind in sorted(SUPPORTED_CHAMPION_ABILITY_KINDS)],
    ]
