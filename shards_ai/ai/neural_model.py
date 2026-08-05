"""Compact action-conditioned PyTorch model for heuristic imitation."""

from __future__ import annotations

from dataclasses import dataclass
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
            nn.Linear(self.config.card_embedding_dim + self.config.action_hidden_dim + 12 + 8 + scalar_action_size,
                      self.config.action_hidden_dim),
            nn.ReLU(),
        )
        # Eleven pooled card groups: active visible/count zones, opponent public
        # aggregates/champions, central deck, and river. Scalars carry resources
        # and masks; the river slot remains available through the action encoder.
        state_input_size = 4 + 4 + 4 + self.config.card_embedding_dim * 11 + 2
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

    def forward(self, observation: NeuralObservation, actions: Sequence[ActionRepresentation]) -> Tensor:
        return self.scores_for_actions(observation, actions)

    def scores_for_actions(self, observation: NeuralObservation, actions: Sequence[ActionRepresentation]) -> Tensor:
        if not actions:
            return torch.empty(0, device=self.device)
        card_ids = list(self._observation_card_ids(observation))
        card_ids.extend(action.card_definition_id for action in actions)
        embedding_lookup = self._embedding_lookup(card_ids)
        state = self.encode_observation(observation, embedding_lookup=embedding_lookup)
        action = self.encode_actions(actions, embedding_lookup=embedding_lookup)
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
        scalars = torch.tensor(
            [active.health / 100, active.mastery / 100, active.gems / 10, active.power / 20,
             opponent.health / 100, opponent.mastery / 100, *[float(value) for value in active.played_faction_mask],
             len(opponent.champions) / 10, len(active.champions) / 10],
            dtype=torch.float32,
            device=self.device,
        )
        context = torch.tensor(
            [self._index(PHASE_TYPES, observation.phase) / max(1, len(PHASE_TYPES) - 1),
             observation.turn_number / 1000], dtype=torch.float32, device=self.device,
        )
        return self.state_encoder(torch.cat((*pools, scalars, context)).unsqueeze(0))

    def encode_actions(
        self,
        actions: Sequence[ActionRepresentation],
        *,
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
        return self.action_encoder(torch.cat((card, action_type, phase, target, scalars), dim=1))

    @property
    def device(self) -> torch.device:
        return self.card_id_embedding.weight.device

    def _card_embedding(self, card_id: str | None) -> Tensor:
        return self._card_embeddings([card_id])[0]

    def _embedding_lookup(self, card_ids: Iterable[str | None]) -> dict[str | None, Tensor]:
        unique_ids = list(dict.fromkeys(card_ids))
        return dict(zip(unique_ids, self._card_embeddings(unique_ids)))

    def _card_embeddings(self, card_ids: Sequence[str | None]) -> Tensor:
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
        action = self.encode_actions(actions, embedding_lookup=embedding_lookup)
        context = self.candidate_context_encoder(action.mean(dim=0, keepdim=True))
        state_batch = state.expand(len(actions), -1)
        context_batch = context.expand(len(actions), -1)
        return self.scorer(torch.cat((state_batch, action, context_batch), dim=1)).squeeze(1)


class SemanticIdentityNeuralActionScorer(NeuralActionScorer):
    """V003 scorer with a wider, explicitly versioned card representation."""

    architecture = "semantic_identity_v3"


SUPPORTED_ARCHITECTURES = (
    NeuralActionScorer.architecture,
    ContextualNeuralActionScorer.architecture,
    SemanticIdentityNeuralActionScorer.architecture,
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
