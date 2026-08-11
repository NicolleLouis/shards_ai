"""Construct neural players for atomic and macro checkpoint contracts."""

from __future__ import annotations

from pathlib import Path

import torch

from shards_ai.game import Game
from shards_ai.game.enums import PlayerId
from shards_ai.game.random import GameRandom

from .neural_model import NeuralActionScorer
from .neural_player import NeuralPlayer
from .neural_training_profiles import load_active_neural_profile


MACRO_ARCHITECTURE_SCHEMA_VERSIONS = {
    "structured_semantic_v5_macro_deck_state_v1": 1,
    "structured_semantic_v5_macro_root_action_v2": 2,
    "structured_semantic_v5_macro_known_consequence_v1": 3,
    "structured_semantic_v5_macro_tactical_action_v1": 4,
}


def is_macro_architecture(architecture: str | None) -> bool:
    return architecture in MACRO_ARCHITECTURE_SCHEMA_VERSIONS


def build_neural_player(
    player_id: PlayerId,
    game: Game,
    rng: GameRandom,
    *,
    checkpoint_path: str | Path | None = None,
    scorer: NeuralActionScorer | None = None,
) -> object:
    """Build the player adapter matching the checkpoint's inference contract."""

    loaded_scorer = scorer
    if loaded_scorer is None:
        selected_checkpoint = checkpoint_path or load_active_neural_profile().checkpoint_path
        loaded_scorer = NeuralPlayer.load_scorer(selected_checkpoint)

    architecture = getattr(loaded_scorer, "architecture", None)
    if not is_macro_architecture(architecture):
        return NeuralPlayer(player_id, None, rng, scorer=loaded_scorer)

    from .macro_player import MacroNeuralPlayer

    def choose_macro(_game: Game, observation, candidates) -> int:
        with torch.inference_mode():
            return int(loaded_scorer(observation, candidates).argmax().item())

    return MacroNeuralPlayer(
        player_id,
        game,
        candidate_scorer=choose_macro,
        candidate_schema_version=MACRO_ARCHITECTURE_SCHEMA_VERSIONS[architecture],
    )
