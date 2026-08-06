"""Structured semantic card encoder for the V004 neural scorer.

This module deliberately leaves the historical scorers in ``neural_model`` untouched.  It reuses
their stable observation/action scorer contract, but owns the card semantic encoder used by V004.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .card_representation import (
    SUPPORTED_CHAMPION_ABILITY_KINDS,
    SUPPORTED_OPERATION_KINDS,
    CardSemanticRepresentation,
    representation_for_definition,
)
from .neural_model import NeuralActionScorer, NeuralModelConfig


_UNK = "<UNK>"
_NONE = "<NONE>"
_FACTIONS = ("maquis", "spectra", "homodeus", "order")
_TARGETS = ("", "self", "opponent", "champion", "card", "other")
_ROLES = ("effect", "on_play", "ability")


def _index(vocabulary: Sequence[str], value: str | None) -> int:
    if value in vocabulary:
        return vocabulary.index(value)
    return vocabulary.index(_UNK) if _UNK in vocabulary else 0


def _normalise(value: int | None, scale: float) -> float:
    return float(value or 0) / scale


class StructuredSemanticCardEncoder(nn.Module):
    """Encode a card definition as masked, ordered semantic tokens."""

    def __init__(
        self,
        config: NeuralModelConfig,
        representations: Mapping[str, CardSemanticRepresentation],
    ) -> None:
        super().__init__()
        hidden = config.semantic_token_hidden_dim
        if hidden % config.semantic_attention_heads != 0:
            raise ValueError("semantic_token_hidden_dim must be divisible by semantic_attention_heads")

        self.config = config
        self.hidden = hidden
        self.operation_kinds = tuple(sorted(SUPPORTED_OPERATION_KINDS)) + (_UNK,)
        self.ability_kinds = tuple(sorted(SUPPORTED_CHAMPION_ABILITY_KINDS)) + (_UNK,)
        passive_values = sorted(
            {
                representation.passive_kind
                for representation in representations.values()
                if representation.passive_kind is not None
            }
        )
        self.passive_kinds = tuple(passive_values) + (_NONE, _UNK)

        self.operation_kind_embedding = nn.Embedding(len(self.operation_kinds), 16)
        self.ability_kind_embedding = nn.Embedding(len(self.ability_kinds), 16)
        self.target_embedding = nn.Embedding(len(_TARGETS), 8)
        self.faction_embedding = nn.Embedding(len(_FACTIONS) + 1, 8)
        self.role_embedding = nn.Embedding(len(_ROLES), 4)
        self.passive_embedding = nn.Embedding(len(self.passive_kinds), 8)

        self.static_encoder = nn.Sequential(
            nn.Linear(12 + 8, hidden),
            nn.ReLU(),
        )
        self.operation_encoder = nn.Sequential(
            nn.Linear(16 + 8 + 8 + 4 + 13, hidden),
            nn.ReLU(),
        )
        self.ability_encoder = nn.Sequential(
            nn.Linear(16 + 8 + 4 + 8, hidden),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden,
            config.semantic_attention_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, config.card_embedding_dim)

    def forward(self, cards: Sequence[CardSemanticRepresentation]) -> Tensor:
        if not cards:
            return torch.empty((0, self.config.card_embedding_dim), device=self.output.weight.device)

        sequences = [self._card_tokens(card) for card in cards]
        max_length = max(sequence.shape[0] for sequence in sequences)
        batch = torch.zeros(
            (len(sequences), max_length, self.hidden),
            dtype=self.output.weight.dtype,
            device=self.output.weight.device,
        )
        padding_mask = torch.ones(
            (len(sequences), max_length), dtype=torch.bool, device=batch.device
        )
        for row, sequence in enumerate(sequences):
            length = sequence.shape[0]
            batch[row, :length] = sequence
            padding_mask[row, :length] = False

        attended, _ = self.attention(batch, batch, batch, key_padding_mask=padding_mask)
        attended = self.attention_norm(batch + attended)
        attended = self.output_norm(attended + self.feed_forward(attended))
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (attended * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.output(pooled)

    def _card_tokens(self, card: CardSemanticRepresentation) -> Tensor:
        static = self._static_token(card)
        tokens = [static]
        position = 0
        for role, effect in (("effect", card.effect), ("on_play", card.on_play_effect)):
            if effect is None:
                continue
            for step_index, step in enumerate(effect.steps):
                for operation in step.operations:
                    tokens.append(self._operation_token(operation, step.mastery_at_least, position, role))
                    position += 1
        if card.champion_ability is not None:
            tokens.append(self._ability_token(card))
        return torch.stack(tokens)

    def _static_token(self, card: CardSemanticRepresentation) -> Tensor:
        faction = [float(card.faction == value) for value in _FACTIONS]
        passive_index = _index(
            self.passive_kinds,
            card.passive_kind if card.passive_kind is not None else _NONE,
        )
        values = torch.tensor(
            [
                card.cost / 20,
                card.shield / 20,
                float(card.is_champion),
                _normalise(card.champion_health, 100),
                float(card.is_mercenary),
                *faction,
                float(card.on_play_effect is not None),
                float(card.champion_ability is not None),
                float(card.passive_kind is not None),
            ],
            dtype=self.output.weight.dtype,
            device=self.output.weight.device,
        )
        passive = self.passive_embedding(
            torch.tensor(passive_index, dtype=torch.long, device=values.device)
        )
        return self.static_encoder(torch.cat((values, passive)))

    def _operation_token(
        self,
        operation,
        step_mastery: int | None,
        position: int,
        role: str,
    ) -> Tensor:
        faction_index = _index(tuple(_FACTIONS) + (_UNK,), operation.faction)
        values = torch.tensor(
            [
                _normalise(operation.amount, 20),
                _normalise(operation.mastery_at_least, 100),
                _normalise(operation.health_at_least, 100),
                _normalise(operation.recruit_to_hand_at_mastery, 100),
                float(operation.mastery_at_least is not None),
                float(operation.health_at_least is not None),
                float(operation.recruit_to_hand_at_mastery is not None),
                float(operation.requires_union),
                float(operation.requires_echo),
                float(operation.requires_domination),
                float(operation.requires_inspiration),
            ],
            dtype=self.output.weight.dtype,
            device=self.output.weight.device,
        )
        categorical = torch.cat(
            (
                self.operation_kind_embedding(
                    torch.tensor(_index(self.operation_kinds, operation.kind), dtype=torch.long, device=values.device)
                ),
                self.target_embedding(
                    torch.tensor(_index(_TARGETS, operation.target), dtype=torch.long, device=values.device)
                ),
                self.faction_embedding(torch.tensor(faction_index, dtype=torch.long, device=values.device)),
                self.role_embedding(
                    torch.tensor(_ROLES.index(role), dtype=torch.long, device=values.device)
                ),
            )
        )
        # Add the step threshold and operation order to the numerical channel.
        values = torch.cat(
            (values, torch.tensor(
                [_normalise(step_mastery, 100), min(position, 20) / 20],
                dtype=values.dtype,
                device=values.device,
            ))
        )
        return self.operation_encoder(torch.cat((categorical, values)))

    def _ability_token(self, card: CardSemanticRepresentation) -> Tensor:
        ability = card.champion_ability
        assert ability is not None
        faction_index = _index(tuple(_FACTIONS) + (_UNK,), ability.faction)
        values = torch.tensor(
            [
                _normalise(ability.amount, 20),
                _normalise(ability.threshold, 100),
                _normalise(ability.secondary_amount, 20),
                _normalise(ability.draw_amount, 10),
                float(ability.threshold is not None),
                float(ability.requires_domination),
                float(card.champion_ability is not None),
                0.0,
            ],
            dtype=self.output.weight.dtype,
            device=self.output.weight.device,
        )
        categorical = torch.cat(
            (
                self.ability_kind_embedding(
                    torch.tensor(_index(self.ability_kinds, ability.kind), dtype=torch.long, device=values.device)
                ),
                self.faction_embedding(torch.tensor(faction_index, dtype=torch.long, device=values.device)),
                self.role_embedding(torch.tensor(2, dtype=torch.long, device=values.device)),
            )
        )
        return self.ability_encoder(torch.cat((categorical, values)))


class StructuredSemanticV4Scorer(NeuralActionScorer):
    """V004 scorer with an ordered, multi-head structured card encoder."""

    architecture = "structured_semantic_v4"

    def __init__(self, config: NeuralModelConfig | None = None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        self.structured_semantic_encoder = StructuredSemanticCardEncoder(
            self.config,
            self._semantic_cache,
        )

    def _card_embeddings_uncached(self, card_ids: Sequence[str | None]) -> Tensor:
        indices = torch.tensor(
            [self.card_to_index.get(card_id or "", self.unk_card_index) for card_id in card_ids],
            dtype=torch.long,
            device=self.device,
        )
        id_vectors = self.card_id_embedding(indices)
        representations = [
            self._semantic_cache.get(card_id, self._semantic_cache[self.card_ids[0]])
            for card_id in card_ids
        ]
        semantic_vectors = self.structured_semantic_encoder(representations)
        return self.card_fusion(torch.cat((id_vectors, semantic_vectors), dim=1))
