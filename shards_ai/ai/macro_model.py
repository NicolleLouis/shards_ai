"""Action-conditioned scorer for bounded PLAY macro candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from shards_ai.game.observation import NeuralObservation

from .action_representation import ActionRepresentation
from .macro_player import MacroActionRepresentation
from .neural_model import ACTION_TYPES, PHASE_TYPES, NeuralModelConfig
from .structured_v005 import StructuredSemanticV5DeckStateScorer


MACRO_ARCHITECTURE = "structured_semantic_v5_macro_deck_state_v1"
MACRO_TERMINAL_TYPES = ("strategic_choice", "phase_end", "game_end")
MACRO_PHASE_TYPES = tuple(value for value in PHASE_TYPES if value)
MACRO_NUMERIC_SIZE = 7
MACRO_CONSEQUENCE_NUMERIC_SIZE = 12
PENDING_TYPES = (
    "destroy_opponent_champion",
    "destroy_all_champions",
    "select_champion_discard",
    "select_mercenary_discard",
    "select_spectra_discard",
    "select_faction_discard",
    "select_effect_copy",
    "recruit_free_card",
    "banish",
)
TRACE_ACTION_TYPE_MAP = {
    "PlayCard": "play_card",
    "ActivateChampion": "activate_champion",
    "BanishCard": "banish_card",
    "SkipBanish": "skip_banish",
    "RecruitFreeCard": "recruit_free_card",
    "PassPlayPhase": "pass_play_phase",
    "GainMastery": "gain_mastery",
    "ChoosePendingDecision": "choose_pending_decision",
}


def macro_candidate_from_dict(value: Mapping[str, object]) -> MacroActionRepresentation:
    """Rebuild a serialized macro representation from a JSONL record."""

    root_value = value.get("root_action")
    root_action = None
    if isinstance(root_value, Mapping):
        root_action = ActionRepresentation(
            action_type=str(root_value["action_type"]),
            phase=str(root_value["phase"]),
            card_definition_id=root_value.get("card_definition_id"),
            river_slot=root_value.get("river_slot"),
            target=root_value.get("target"),
            amount=root_value.get("amount"),
            card_instance_id=None,
            choice_id=None,
        )
    return MacroActionRepresentation(
        schema_version=int(value["schema_version"]),
        decision_kind=str(value.get("decision_kind", "macro_play")),
        action_type=str(value["action_type"]),
        trace_action_types=tuple(str(item) for item in value["trace_action_types"]),
        terminal_kind=str(value["terminal_kind"]),
        phase=str(value["phase"]),
        gems=int(value["gems"]),
        mastery=int(value["mastery"]),
        power=int(value["power"]),
        hand_size=int(value["hand_size"]),
        discard_size=int(value["discard_size"]),
        play_zone_size=int(value["play_zone_size"]),
        atomic_action_count=int(value["atomic_action_count"]),
        physical_variant_count=int(value.get("physical_variant_count", 1)),
        root_action=root_action,
        delta_gems=int(value.get("delta_gems", 0)),
        delta_mastery=int(value.get("delta_mastery", 0)),
        delta_power=int(value.get("delta_power", 0)),
        delta_active_health=int(value.get("delta_active_health", 0)),
        delta_opponent_health=int(value.get("delta_opponent_health", 0)),
        delta_hand_size=int(value.get("delta_hand_size", 0)),
        delta_discard_size=int(value.get("delta_discard_size", 0)),
        delta_play_zone_size=int(value.get("delta_play_zone_size", 0)),
        delta_active_champion_count=int(value.get("delta_active_champion_count", 0)),
        delta_opponent_champion_count=int(value.get("delta_opponent_champion_count", 0)),
        pending_kind=value.get("pending_kind"),
        pending_choice_count=int(value.get("pending_choice_count", 0)),
        known_card_definition_ids=tuple(value.get("known_card_definition_ids", ())),
        played_faction_mask=tuple(value.get("played_faction_mask", ())),
        played_champion_faction_mask=tuple(value.get("played_champion_faction_mask", ())),
        immediate_victory=bool(value.get("immediate_victory", False)),
        requires_union=bool(value.get("requires_union", False)),
        union_active=bool(value.get("union_active", False)),
        requires_echo=bool(value.get("requires_echo", False)),
        echo_active=bool(value.get("echo_active", False)),
        requires_domination=bool(value.get("requires_domination", False)),
        domination_active=bool(value.get("domination_active", False)),
        domination_missing_count=float(value.get("domination_missing_count", 0.0)),
    )


class MacroActionScorer(StructuredSemanticV5DeckStateScorer):
    """Score a variable number of solver consequences using deck-state features."""

    architecture = MACRO_ARCHITECTURE

    def __init__(self, config: NeuralModelConfig | None = None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        if self.config.observation_feature_set != "deck_state_v1":
            raise ValueError("MacroActionScorer requires observation_feature_set='deck_state_v1'")
        macro_input_size = len(ACTION_TYPES) + len(MACRO_TERMINAL_TYPES) + len(MACRO_PHASE_TYPES) + MACRO_NUMERIC_SIZE
        self.macro_action_encoder = nn.Sequential(
            nn.Linear(macro_input_size, self.config.action_hidden_dim),
            nn.ReLU(),
        )
        self.macro_scorer = nn.Sequential(
            nn.Linear(self.config.state_hidden_dim + self.config.action_hidden_dim, self.config.scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.scorer_hidden_dim, 1),
        )

    def forward(
        self,
        observation: NeuralObservation,
        candidates: Sequence[MacroActionRepresentation],
    ) -> Tensor:
        if not candidates:
            return torch.empty(0, device=self.device)
        state = self.encode_observation(observation)
        candidate = self.encode_macro_candidates(candidates)
        state_batch = state.expand(len(candidates), -1)
        return self.macro_scorer(torch.cat((state_batch, candidate), dim=1)).squeeze(1)

    def encode_macro_candidates(self, candidates: Sequence[MacroActionRepresentation]) -> Tensor:
        values = self._macro_feature_tensor(candidates)
        return self.macro_action_encoder(values)

    def _macro_feature_tensor(self, candidates: Sequence[MacroActionRepresentation]) -> Tensor:
        rows = []
        for candidate_value in candidates:
            candidate = getattr(candidate_value, "representation", candidate_value)
            normalized_trace = [
                TRACE_ACTION_TYPE_MAP.get(action_type, action_type)
                for action_type in candidate.trace_action_types
            ]
            trace_counts = Counter(normalized_trace)
            action_counts = [trace_counts[action_type] for action_type in ACTION_TYPES]
            terminal = [float(candidate.terminal_kind == value) for value in MACRO_TERMINAL_TYPES]
            phase = [float(candidate.phase == value) for value in MACRO_PHASE_TYPES]
            numeric = [
                candidate.gems / 10,
                candidate.mastery / 100,
                candidate.power / 20,
                candidate.hand_size / 20,
                candidate.discard_size / 100,
                candidate.play_zone_size / 20,
                candidate.atomic_action_count / 32,
            ]
            rows.append([*action_counts, *terminal, *phase, *numeric])
        values = torch.tensor(rows, dtype=self.card_id_embedding.weight.dtype, device=self.device)
        return values


MACRO_V2_ARCHITECTURE = "structured_semantic_v5_macro_root_action_v2"


class MacroActionScorerV2(MacroActionScorer):
    """V2 scorer adding the masked root action to the candidate contract."""

    architecture = MACRO_V2_ARCHITECTURE

    def __init__(self, config: NeuralModelConfig | None = None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        v1_size = len(ACTION_TYPES) + len(MACRO_TERMINAL_TYPES) + len(MACRO_PHASE_TYPES) + MACRO_NUMERIC_SIZE
        self.macro_action_encoder = nn.Sequential(
            nn.Linear(v1_size + self.config.action_hidden_dim, self.config.action_hidden_dim),
            nn.ReLU(),
        )

    def encode_macro_candidates(
        self,
        candidates: Sequence[MacroActionRepresentation],
        *,
        observation: NeuralObservation | None = None,
        embedding_lookup: dict[str | None, Tensor] | None = None,
    ) -> Tensor:
        if observation is None:
            raise ValueError("MacroActionScorerV2 requires the decision observation")
        base = self._macro_feature_tensor(candidates)
        roots = [getattr(candidate, "representation", candidate).root_action for candidate in candidates]
        if any(root is None for root in roots):
            raise ValueError("Macro V2 candidates require root_action")
        root_encoded = self.encode_actions(
            [root for root in roots if root is not None],
            observation=observation, embedding_lookup=embedding_lookup,
        )
        return self.macro_action_encoder(torch.cat((base, root_encoded), dim=1))

    def forward(
        self,
        observation: NeuralObservation,
        candidates: Sequence[MacroActionRepresentation],
    ) -> Tensor:
        if not candidates:
            return torch.empty(0, device=self.device)
        state = self.encode_observation(observation)
        candidate = self.encode_macro_candidates(candidates, observation=observation)
        state_batch = state.expand(len(candidates), -1)
        return self.macro_scorer(torch.cat((state_batch, candidate), dim=1)).squeeze(1)


MACRO_V3_ARCHITECTURE = "structured_semantic_v5_macro_known_consequence_v1"


class MacroActionScorerV3(MacroActionScorerV2):
    """V2 scorer with bounded, publicly known branch consequences."""

    architecture = MACRO_V3_ARCHITECTURE

    def __init__(self, config: NeuralModelConfig | None = None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        v1_size = len(ACTION_TYPES) + len(MACRO_TERMINAL_TYPES) + len(MACRO_PHASE_TYPES) + MACRO_NUMERIC_SIZE
        consequence_size = (
            MACRO_CONSEQUENCE_NUMERIC_SIZE
            + len(PENDING_TYPES)
            + 8
            + self.config.card_embedding_dim
        )
        self.macro_action_encoder = nn.Sequential(
            nn.Linear(v1_size + self.config.action_hidden_dim + consequence_size, self.config.action_hidden_dim),
            nn.ReLU(),
        )

    def encode_macro_candidates(
        self,
        candidates: Sequence[MacroActionRepresentation],
        *,
        observation: NeuralObservation | None = None,
        embedding_lookup: dict[str | None, Tensor] | None = None,
    ) -> Tensor:
        if observation is None:
            raise ValueError("MacroActionScorerV3 requires the decision observation")
        base = self._macro_feature_tensor(candidates)
        roots = [getattr(candidate, "representation", candidate).root_action for candidate in candidates]
        if any(root is None for root in roots):
            raise ValueError("Macro V3 candidates require root_action")
        root_encoded = self.encode_actions(
            [root for root in roots if root is not None],
            observation=observation, embedding_lookup=embedding_lookup,
        )
        consequence = self._known_consequence_tensor(candidates)
        return self.macro_action_encoder(torch.cat((base, root_encoded, consequence), dim=1))

    def _known_consequence_tensor(self, candidates: Sequence[MacroActionRepresentation]) -> Tensor:
        rows = []
        for candidate in candidates:
            candidate = getattr(candidate, "representation", candidate)
            deltas = [
                candidate.delta_gems / 10,
                candidate.delta_mastery / 100,
                candidate.delta_power / 20,
                candidate.delta_active_health / 100,
                candidate.delta_opponent_health / 100,
                candidate.delta_hand_size / 20,
                candidate.delta_discard_size / 20,
                candidate.delta_play_zone_size / 20,
                candidate.delta_active_champion_count / 10,
                candidate.delta_opponent_champion_count / 10,
                min(max(candidate.pending_choice_count, 0), 16) / 16,
                float(candidate.immediate_victory),
            ]
            pending = [float(candidate.pending_kind == value) for value in PENDING_TYPES]
            masks = [
                float(value) for value in (*candidate.played_faction_mask, *candidate.played_champion_faction_mask)
            ]
            masks.extend([0.0] * (8 - len(masks)))
            known = self._known_card_embedding(candidate.known_card_definition_ids)
            rows.append(torch.cat((
                torch.tensor([*deltas, *pending, *masks], dtype=known.dtype, device=self.device),
                known,
            )))
        return torch.stack(rows) if rows else torch.empty((0, MACRO_CONSEQUENCE_NUMERIC_SIZE + len(PENDING_TYPES) + 8 + self.config.card_embedding_dim), device=self.device)

    def _known_card_embedding(self, card_ids: Sequence[str]) -> Tensor:
        if not card_ids:
            return torch.zeros(self.config.card_embedding_dim, dtype=self.card_id_embedding.weight.dtype, device=self.device)
        return self._card_embeddings(card_ids).mean(dim=0)

    def forward(self, observation: NeuralObservation, candidates: Sequence[MacroActionRepresentation]) -> Tensor:
        if not candidates:
            return torch.empty(0, device=self.device)
        state = self.encode_observation(observation)
        candidate = self.encode_macro_candidates(candidates, observation=observation)
        state_batch = state.expand(len(candidates), -1)
        return self.macro_scorer(torch.cat((state_batch, candidate), dim=1)).squeeze(1)


MACRO_V4_ARCHITECTURE = "structured_semantic_v5_macro_tactical_action_v1"


class MacroActionScorerV4(MacroActionScorerV3):
    """V3 macro scorer with action-conditioned Union/Echo/Domination features."""

    architecture = MACRO_V4_ARCHITECTURE

    def __init__(self, config: NeuralModelConfig | None = None, *, card_catalog=None) -> None:
        super().__init__(config, card_catalog=card_catalog)
        v1_size = len(ACTION_TYPES) + len(MACRO_TERMINAL_TYPES) + len(MACRO_PHASE_TYPES) + MACRO_NUMERIC_SIZE
        consequence_size = (
            MACRO_CONSEQUENCE_NUMERIC_SIZE
            + len(PENDING_TYPES)
            + 8
            + self.config.card_embedding_dim
        )
        self.macro_action_encoder = nn.Sequential(
            nn.Linear(
                v1_size + self.config.action_hidden_dim + consequence_size + 7 + 2,
                self.config.action_hidden_dim,
            ),
            nn.ReLU(),
        )

    def encode_macro_candidates(
        self,
        candidates: Sequence[MacroActionRepresentation],
        *,
        observation: NeuralObservation | None = None,
        embedding_lookup: dict[str | None, Tensor] | None = None,
    ) -> Tensor:
        if observation is None:
            raise ValueError("MacroActionScorerV4 requires the decision observation")
        base = self._macro_feature_tensor(candidates)
        roots = [getattr(candidate, "representation", candidate).root_action for candidate in candidates]
        if any(root is None for root in roots):
            raise ValueError("Macro V4 candidates require root_action")
        root_encoded = self.encode_actions(
            [root for root in roots if root is not None],
            observation=observation, embedding_lookup=embedding_lookup,
        )
        consequence = self._known_consequence_tensor(candidates)
        tactical = torch.tensor(
            [[
                float(getattr(candidate, "representation", candidate).requires_union),
                float(getattr(candidate, "representation", candidate).union_active),
                float(getattr(candidate, "representation", candidate).requires_echo),
                float(getattr(candidate, "representation", candidate).echo_active),
                float(getattr(candidate, "representation", candidate).requires_domination),
                float(getattr(candidate, "representation", candidate).domination_active),
                float(getattr(candidate, "representation", candidate).domination_missing_count),
            ] for candidate in candidates],
            dtype=self.card_id_embedding.weight.dtype,
            device=self.device,
        )
        decision_kind = torch.tensor(
            [[float(getattr(candidate, "representation", candidate).decision_kind == "macro_play"), float(getattr(candidate, "representation", candidate).decision_kind == "atomic")]
             for candidate in candidates],
            dtype=self.card_id_embedding.weight.dtype,
            device=self.device,
        )
        return self.macro_action_encoder(torch.cat((base, root_encoded, consequence, tactical, decision_kind), dim=1))




__all__ = [
    "MACRO_ARCHITECTURE",
    "MACRO_V2_ARCHITECTURE",
    "MACRO_V3_ARCHITECTURE",
    "MACRO_V4_ARCHITECTURE",
    "MacroActionScorer",
    "MacroActionScorerV2",
    "MacroActionScorerV3",
    "MacroActionScorerV4",
    "macro_candidate_from_dict",
]
