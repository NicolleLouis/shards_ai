import torch
import pytest

from dataclasses import asdict, replace

from shards_ai.ai import (
    MACRO_V2_ARCHITECTURE,
    MACRO_V3_ARCHITECTURE,
    MACRO_V4_ARCHITECTURE,
    MacroActionScorer,
    MacroActionScorerV2,
    MacroActionScorerV3,
    MacroActionScorerV4,
    NeuralModelConfig,
    PlayTurnSolver,
    macro_candidate_from_dict,
)
from shards_ai.ai.action_representation import ActionRepresentation
from shards_ai.ai.play_turn_solver import macro_representations_v2, macro_representations_v3, macro_representations_v4
from shards_ai.ai.structured_v006 import StructuredSemanticV6TacticalActionScorer
from shards_ai.game import CARD_CATALOG, Game
from shards_ai.game.actions import PlayCard
from shards_ai.game.cards.definitions import SHARD_REACTOR


def test_macro_scorer_uses_deck_state_features_and_variable_candidates() -> None:
    game = Game.new(seed=1510)
    resolution = PlayTurnSolver().resolve(game)
    model = MacroActionScorer(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )
    observation = resolution.observation_game.neural_observation_for(game.active_player)

    scores = model(observation, resolution.candidates[:2])

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert model._observation_feature_size() == 11


def test_macro_v2_encodes_root_action_without_instance_identity() -> None:
    game = Game.new(seed=1512)
    resolution = PlayTurnSolver().resolve(game)
    observation = resolution.observation_game.neural_observation_for(game.active_player)
    representations = macro_representations_v2(observation, resolution.candidates[:2])
    assert all(item.schema_version == 2 for item in representations)
    assert all(item.root_action is not None for item in representations)
    assert representations[0].root_action.card_instance_id is None
    assert representations[0].root_action.choice_id is None
    assert representations[0].root_action != representations[1].root_action

    model = MacroActionScorerV2(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )
    scores = model(observation, representations)

    assert model.architecture == MACRO_V2_ARCHITECTURE
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()

    restored = macro_candidate_from_dict(asdict(representations[0]))
    assert restored.root_action == representations[0].root_action


def test_macro_v2_root_encoding_is_instance_invariant() -> None:
    game = Game.new(seed=1513)
    resolution = PlayTurnSolver().resolve(game)
    observation = resolution.observation_game.neural_observation_for(game.active_player)
    root = macro_representations_v2(observation, resolution.candidates[:1])[0].root_action
    assert root is not None
    first = replace(root, card_instance_id="instance-a", choice_id="choice-a")
    second = replace(root, card_instance_id="instance-b", choice_id="choice-b")
    model = MacroActionScorerV2(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )

    encoded = model.encode_actions((first, second), observation=observation)

    torch.testing.assert_close(encoded[0], encoded[1])


def test_macro_v2_root_encoding_distinguishes_card_definitions() -> None:
    game = Game.new(seed=1514)
    observation = game.neural_observation_for(game.active_player)
    card_a, card_b = sorted(CARD_CATALOG)[:2]
    model = MacroActionScorerV2(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )
    roots = (
        ActionRepresentation("play_card", "play", card_definition_id=card_a),
        ActionRepresentation("play_card", "play", card_definition_id=card_b),
        ActionRepresentation("banish_card", "play", card_definition_id=card_a),
        ActionRepresentation("banish_card", "play", card_definition_id=card_b),
    )

    encoded = model.encode_actions(roots, observation=observation)

    assert not torch.equal(encoded[0], encoded[1])
    assert not torch.equal(encoded[2], encoded[3])


def test_macro_v2_cannot_resolve_a_hidden_root_card() -> None:
    game = Game.new(seed=1515)
    resolution = PlayTurnSolver().resolve(game)
    observation = resolution.observation_game.neural_observation_for(game.active_player)
    hidden_candidate = replace(
        resolution.candidates[0],
        atomic_trace=(PlayCard("card-only-in-hidden-zone"),),
    )

    with pytest.raises(ValueError, match="public card|visible"):
        macro_representations_v2(observation, (hidden_candidate,))


def test_macro_v3_encodes_known_consequences_and_masks() -> None:
    game = Game.new(seed=1516, card_definition=SHARD_REACTOR)
    resolution = PlayTurnSolver().resolve(game)
    observation = resolution.observation_game.neural_observation_for(game.active_player)
    representations = macro_representations_v3(observation, resolution.candidates[:2])

    assert all(item.schema_version == 3 for item in representations)
    assert any(item.known_card_definition_ids for item in representations)
    assert all(len(item.played_faction_mask) == 4 for item in representations)
    assert all(len(item.played_champion_faction_mask) == 4 for item in representations)

    model = MacroActionScorerV3(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )
    encoded = model.encode_macro_candidates(representations, observation=observation)
    scores = model(observation, representations)

    assert model.architecture == MACRO_V3_ARCHITECTURE
    assert encoded.shape == (2, 16)
    assert torch.isfinite(scores).all()


def test_macro_v4_reuses_v6_tactical_features_per_root_action() -> None:
    game = Game.new(seed=1517, card_definition=SHARD_REACTOR)
    resolution = PlayTurnSolver().resolve(game)
    observation = resolution.observation_game.neural_observation_for(game.active_player)
    representations = macro_representations_v4(observation, resolution.candidates[:4])
    config = replace(
        NeuralModelConfig(),
        observation_feature_set="deck_state_v1",
        card_embedding_dim=16,
        semantic_hidden_dim=24,
        state_hidden_dim=32,
        action_hidden_dim=16,
        scorer_hidden_dim=24,
        semantic_token_hidden_dim=32,
    )
    macro_model = MacroActionScorerV4(config)
    v6_model = StructuredSemanticV6TacticalActionScorer(config)

    for candidate, representation in zip(representations, resolution.candidates[:4]):
        root = representation.atomic_trace[0]
        root_representation = __import__(
            "shards_ai.ai.action_representation", fromlist=["representation_for_neural_action"]
        ).representation_for_neural_action(root, observation)
        expected = v6_model._features_for_action(observation, root_representation)
        actual = (
            float(candidate.requires_union), float(candidate.union_active),
            float(candidate.requires_echo), float(candidate.echo_active),
            float(candidate.requires_domination), float(candidate.domination_active),
            candidate.domination_missing_count,
        )
        assert actual == expected

    scores = macro_model(observation, representations)
    assert macro_model.architecture == MACRO_V4_ARCHITECTURE
    assert torch.isfinite(scores).all()
    assert representations[0].root_action is not None
