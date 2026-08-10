"""V005 card-fusion experiments built on the structured V004 encoder."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .structured_v004 import StructuredSemanticV4Scorer


class StructuredSemanticV5FusionScorer(StructuredSemanticV4Scorer):
    """Experiment-only scorer for normalized card-fusion scales.

    The V004 scorer remains the historical/raw-fusion implementation.  V005 keeps
    the same encoder and scorer modules but applies its explicit fusion contract
    before the shared ``card_fusion`` layer.
    """

    architecture = "structured_semantic_v5_fusion_experiment"

    def __init__(self, config=None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        self._validate_fusion_config()

    def _validate_fusion_config(self) -> None:
        config = self.config
        for name in ("card_fusion_id_scale", "card_fusion_semantic_scale"):
            value = float(getattr(config, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if config.card_fusion_normalization != "l2":
            raise ValueError("V005 currently supports only card_fusion_normalization='l2'")
        epsilon = float(config.card_fusion_normalization_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("card_fusion_normalization_epsilon must be a finite positive number")

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
        id_vectors = self._normalize(id_vectors) * self.config.card_fusion_id_scale
        semantic_vectors = self._normalize(semantic_vectors) * self.config.card_fusion_semantic_scale
        return self.card_fusion(torch.cat((id_vectors, semantic_vectors), dim=1))

    def _normalize(self, vectors: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(vectors, ord=2, dim=1, keepdim=True)
        return vectors / norms.clamp_min(self.config.card_fusion_normalization_epsilon)


class StructuredSemanticV5DeckStateScorer(StructuredSemanticV5FusionScorer):
    """V005 scorer with the combined deck cardinality/faction features."""

    architecture = "structured_semantic_v5_deck_state_v1"

    def __init__(self, config=None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        if self.config.observation_feature_set != "deck_state_v1":
            raise ValueError(
                "StructuredSemanticV5DeckStateScorer requires "
                "observation_feature_set='deck_state_v1'"
            )
