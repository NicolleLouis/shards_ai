from __future__ import annotations

import json
from dataclasses import asdict, replace

import torch

from shards_ai.ai import (
    ContextualNeuralActionScorer,
    NeuralActionScorer,
    NeuralModelConfig,
    SemanticIdentityNeuralActionScorer,
    StructuredSemanticCardEncoder,
    StructuredSemanticV5DeckStateScorer,
    StructuredSemanticV4Scorer,
    StructuredSemanticV5FusionScorer,
    StructuredSemanticV6TacticalActionScorer,
    build_neural_scorer,
    migrate_v004_checkpoint_to_deck_state,
    migrate_v005_deck_state_checkpoint_to_tactical,
    representation_for_definition,
    representation_for_action,
)
from shards_ai.ai.neural_training import (
    chosen_action_loss,
    combined_imitation_loss,
    observation_from_dict,
    pairwise_ranking_loss,
    train_epoch,
)
from shards_ai.game import CardDefinition, CardInstance, Effect, EffectStep, Game, Operation, card_definition
from shards_ai.game.actions import PlayCard


def _decision_fixture() -> tuple[Game, object, list]:
    game = Game.new(seed=901)
    observation = game.neural_observation_for(game.active_player)
    representations = [representation_for_action(action, game.state) for action in game.legal_actions()]
    return game, observation, representations


def test_action_conditioned_model_scores_all_legal_actions() -> None:
    _game, observation, actions = _decision_fixture()
    model = NeuralActionScorer(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))

    scores = model(observation, actions)

    assert scores.shape == (len(actions),)
    assert torch.isfinite(scores).all()


def test_observation_feature_set_preserves_historical_state_dimension() -> None:
    config = NeuralModelConfig(card_embedding_dim=16)
    baseline = NeuralActionScorer(config)
    cardinality = NeuralActionScorer(
        replace(config, observation_feature_set="zone_cardinality_v1")
    )

    assert baseline.state_encoder[0].in_features == 11 * 16 + 12 + 2
    assert cardinality.state_encoder[0].in_features == 11 * 16 + 12 + 7 + 2


def test_zone_cardinality_features_distinguish_same_card_pool_size() -> None:
    _game, observation, _actions = _decision_fixture()
    card_id = observation.active_player.hand[0].card_definition_id
    one_card = replace(observation.active_player, draw_pile_counts=((card_id, 1),))
    two_cards = replace(observation.active_player, draw_pile_counts=((card_id, 2),))
    one_observation = replace(observation, active_player=one_card)
    two_observation = replace(observation, active_player=two_cards)
    model = NeuralActionScorer(
        NeuralModelConfig(observation_feature_set="zone_cardinality_v1")
    )

    one_scalars = model._state_scalars(one_observation)
    two_scalars = model._state_scalars(two_observation)

    assert one_scalars[12] != two_scalars[12]
    assert torch.isfinite(one_scalars).all()
    assert torch.isfinite(two_scalars).all()


def test_unknown_observation_feature_set_is_rejected() -> None:
    try:
        NeuralActionScorer(NeuralModelConfig(observation_feature_set="unknown"))
    except ValueError as error:
        assert "observation_feature_set" in str(error)
    else:
        raise AssertionError("unknown observation feature set should be rejected")


def test_contextual_model_scores_candidates_and_is_order_equivariant() -> None:
    game, observation, actions = _decision_fixture()
    model = ContextualNeuralActionScorer(
        NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16, candidate_context_dim=8)
    )

    scores = model(observation, actions)
    reversed_scores = model(observation, list(reversed(actions)))

    assert scores.shape == (len(actions),)
    assert torch.allclose(scores, torch.flip(reversed_scores, dims=(0,)))


def test_v003_semantic_identity_model_scores_candidates_and_is_order_equivariant() -> None:
    game, observation, actions = _decision_fixture()
    config = NeuralModelConfig(card_id_embedding_dim=24, semantic_hidden_dim=64, card_embedding_dim=64)
    model = build_neural_scorer("semantic_identity_v3", config)

    scores = model(observation, actions)
    reversed_scores = model(observation, list(reversed(actions)))

    assert isinstance(model, SemanticIdentityNeuralActionScorer)
    assert model._card_embedding(actions[0].card_definition_id).shape == (64,)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.flip(reversed_scores, dims=(0,)))


def test_v004_structured_model_scores_candidates_and_is_order_equivariant() -> None:
    game, observation, actions = _decision_fixture()
    config = NeuralModelConfig(
        card_embedding_dim=32,
        state_hidden_dim=32,
        action_hidden_dim=16,
        semantic_token_hidden_dim=32,
        semantic_attention_heads=4,
    )
    model = build_neural_scorer("structured_semantic_v4", config)

    scores = model(observation, actions)
    reversed_scores = model(observation, list(reversed(actions)))

    assert isinstance(model, StructuredSemanticV4Scorer)
    assert model._card_embedding(actions[0].card_definition_id).shape == (32,)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.flip(reversed_scores, dims=(0,)))


def test_v005_fusion_experiment_scores_candidates_and_is_order_equivariant() -> None:
    game, observation, actions = _decision_fixture()
    config = NeuralModelConfig(
        card_embedding_dim=32,
        state_hidden_dim=32,
        action_hidden_dim=16,
        semantic_token_hidden_dim=32,
        semantic_attention_heads=4,
    )
    model = build_neural_scorer("structured_semantic_v5_fusion_experiment", config)

    scores = model(observation, actions)
    reversed_scores = model(observation, list(reversed(actions)))

    assert isinstance(model, StructuredSemanticV5FusionScorer)
    assert model._card_embedding(actions[0].card_definition_id).shape == (32,)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.flip(reversed_scores, dims=(0,)))


def test_v005_deck_state_model_scores_candidates_and_has_combined_features() -> None:
    game, observation, actions = _decision_fixture()
    config = NeuralModelConfig(
        card_embedding_dim=32,
        state_hidden_dim=32,
        action_hidden_dim=16,
        semantic_token_hidden_dim=32,
        semantic_attention_heads=4,
        observation_feature_set="deck_state_v1",
    )
    model = build_neural_scorer("structured_semantic_v5_deck_state_v1", config)

    scores = model(observation, actions)

    assert isinstance(model, StructuredSemanticV5DeckStateScorer)
    assert model.state_encoder[0].in_features == 11 * 32 + 12 + 11 + 2
    assert torch.isfinite(scores).all()


def test_v006_tactical_model_scores_actions_with_candidate_features() -> None:
    game = Game.new(seed=902)
    candidate = CardInstance("candidate", card_definition("aspirant_maquis"))
    ally = CardInstance("ally", card_definition("chevalier_le_shai"))
    game.active.hand = [candidate, ally]
    observation = game.neural_observation_for(game.active_player)
    action = representation_for_action(PlayCard(candidate.instance_id), game.state)
    config = NeuralModelConfig(
        card_embedding_dim=32,
        state_hidden_dim=32,
        action_hidden_dim=16,
        semantic_token_hidden_dim=32,
        semantic_attention_heads=4,
        observation_feature_set="deck_state_v1",
    )
    model = build_neural_scorer("structured_semantic_v6_tactical_action_v1", config)

    features = model._features_for_action(observation, action)
    scores = model(observation, [action])

    assert isinstance(model, StructuredSemanticV6TacticalActionScorer)
    assert model.action_encoder[0].in_features == 32 + 16 + 12 + 8 + 5 + 7
    assert features[:2] == (1.0, 1.0)
    assert torch.isfinite(scores).all()


def test_tactical_features_follow_engine_zones_and_played_champions() -> None:
    game = Game.new(seed=903)
    union_candidate = CardInstance("union-candidate", card_definition("aspirant_maquis"))
    domination_candidate = CardInstance("domination-candidate", card_definition("omnius_l_erudit"))
    maquis = CardInstance("maquis", card_definition("chevalier_le_shai"))
    homodeus_champion = CardInstance("homodeus-champion", card_definition("drakonarius"))
    game.active.hand = [union_candidate, domination_candidate, maquis]
    game.active.play_zone = [CardInstance("spectra", card_definition("eclaireur_spectral"))]
    game.active.champions = [homodeus_champion]
    game.active.played_card_ids_this_turn = {homodeus_champion.instance_id}
    game.active.activated_champion_ids = set()
    game.active.discard_pile = [CardInstance("discard", card_definition("eclaireur_spectral"))]
    observation = game.neural_observation_for(game.active_player)
    model = build_neural_scorer(
        "structured_semantic_v6_tactical_action_v1",
        NeuralModelConfig(observation_feature_set="deck_state_v1"),
    )

    union_action = representation_for_action(PlayCard(union_candidate.instance_id), game.state)
    domination_action = representation_for_action(PlayCard(domination_candidate.instance_id), game.state)
    union_features = model._features_for_action(observation, union_action)
    domination_features = model._features_for_action(observation, domination_action)

    assert union_features[:2] == (1.0, 1.0)
    assert domination_features[4:6] == (1.0, 1.0)
    assert domination_features[6] == 0.0


def test_tactical_features_are_neutral_for_non_play_actions() -> None:
    game, observation, actions = _decision_fixture()
    model = build_neural_scorer(
        "structured_semantic_v6_tactical_action_v1",
        NeuralModelConfig(observation_feature_set="deck_state_v1"),
    )

    features = [model._features_for_action(observation, action) for action in actions]

    assert all(feature == (0.0,) * 7 for action, feature in zip(actions, features) if action.action_type != "play_card")


def test_v005_checkpoint_migration_expands_action_encoder() -> None:
    config = NeuralModelConfig(observation_feature_set="deck_state_v1", card_embedding_dim=16, action_hidden_dim=16)
    source = build_neural_scorer("structured_semantic_v5_deck_state_v1", config)
    checkpoint = {
        "architecture": "structured_semantic_v5_deck_state_v1",
        "model_config": asdict(config),
        "model_state_dict": source.state_dict(),
    }

    migrated = migrate_v005_deck_state_checkpoint_to_tactical(checkpoint)
    target = build_neural_scorer(migrated["architecture"], NeuralModelConfig(**migrated["model_config"]))
    target.load_state_dict(migrated["model_state_dict"])

    source_weight = checkpoint["model_state_dict"]["action_encoder.0.weight"]
    target_weight = migrated["model_state_dict"]["action_encoder.0.weight"]
    assert target_weight.shape[1] == source_weight.shape[1] + 7
    assert torch.equal(target_weight[:, :source_weight.shape[1]], source_weight)
    assert torch.count_nonzero(target_weight[:, source_weight.shape[1]:]) == 0


def test_deck_state_features_count_factions_and_ignore_neutrals() -> None:
    _game, observation, _actions = _decision_fixture()
    active = replace(
        observation.active_player,
        owned_card_counts=(
            ("aspirant_maquis", 2),
            ("eclaireur_spectral", 3),
            ("crystal", 7),
        ),
    )
    model = NeuralActionScorer(
        NeuralModelConfig(observation_feature_set="deck_state_v1")
    )

    scalars = model._state_scalars(replace(observation, active_player=active))

    assert torch.allclose(scalars[19:], torch.tensor([0.02, 0.03, 0.0, 0.0]))


def test_v004_checkpoint_migration_expands_state_encoder_and_metadata() -> None:
    config = NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16)
    source = build_neural_scorer("structured_semantic_v5_fusion_experiment", config)
    checkpoint = {
        "architecture": "structured_semantic_v5_fusion_experiment",
        "model_config": asdict(config),
        "model_state_dict": source.state_dict(),
    }

    migrated = migrate_v004_checkpoint_to_deck_state(checkpoint)
    target_config = NeuralModelConfig(**migrated["model_config"])
    target = build_neural_scorer(
        migrated["architecture"],
        target_config,
    )
    target.load_state_dict(migrated["model_state_dict"])

    source_weight = checkpoint["model_state_dict"]["state_encoder.0.weight"]
    target_weight = migrated["model_state_dict"]["state_encoder.0.weight"]
    prefix_size = source_weight.shape[1] - 2
    assert target_weight.shape[1] == source_weight.shape[1] + 11
    assert torch.equal(target_weight[:, :prefix_size], source_weight[:, :prefix_size])
    assert torch.equal(target_weight[:, prefix_size + 11:], source_weight[:, prefix_size:])
    assert torch.count_nonzero(target_weight[:, prefix_size:prefix_size + 11]) == 0
    assert migrated["migration"]["target_observation_feature_set"] == "deck_state_v1"


def test_v005_rejects_invalid_fusion_configuration() -> None:
    config = NeuralModelConfig(card_fusion_normalization="none")

    try:
        StructuredSemanticV5FusionScorer(config)
    except ValueError as error:
        assert "card_fusion_normalization" in str(error)
    else:
        raise AssertionError("invalid V005 normalization should be rejected")


def test_v004_semantic_encoder_keeps_operation_amounts() -> None:
    one_draw = CardDefinition(
        card_id="one-draw",
        name="One draw",
        cost=1,
        effect=Effect(steps=(EffectStep((Operation("draw_card", amount=1),)),)),
    )
    three_draws = CardDefinition(
        card_id="three-draws",
        name="Three draws",
        cost=1,
        effect=Effect(steps=(EffectStep((Operation("draw_card", amount=3),)),)),
    )
    config = NeuralModelConfig(semantic_token_hidden_dim=32, semantic_attention_heads=4)
    encoder = StructuredSemanticCardEncoder(
        config,
        {
            card.card_id: representation_for_definition(card)
            for card in (one_draw, three_draws)
        },
    )

    embeddings = encoder(
        [representation_for_definition(one_draw), representation_for_definition(three_draws)]
    )

    assert embeddings.shape == (2, config.card_embedding_dim)
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_model_accepts_serialized_masked_observation() -> None:
    _game, observation, actions = _decision_fixture()
    restored = observation_from_dict(json.loads(json.dumps(asdict(observation))))
    model = NeuralActionScorer()

    assert model(restored, actions).shape == (len(actions),)


def test_ranking_loss_only_uses_strict_teacher_order() -> None:
    predicted = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss = pairwise_ranking_loss(predicted, torch.tensor([2.0, 1.0, 1.0]))

    assert loss.item() > 0
    loss.backward()
    assert predicted.grad is not None

    assert pairwise_ranking_loss(predicted.detach(), torch.tensor([1.0, 1.0, 1.0])).item() == 0


def test_combined_loss_can_train_one_json_record() -> None:
    game, observation, actions = _decision_fixture()
    model = NeuralActionScorer(NeuralModelConfig(card_embedding_dim=16, state_hidden_dim=32, action_hidden_dim=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    record = {
        "observation": asdict(observation),
        "action_representations": [action.to_dict() for action in actions],
        "heuristic_scores": list(range(len(actions), 0, -1)),
        "chosen_action_index": 0,
    }

    metrics = train_epoch(model, [record], optimizer)

    assert metrics.records == 1
    assert metrics.mean_loss >= 0
    assert chosen_action_loss(torch.tensor([1.0, 0.0]), 0).item() > 0
    assert combined_imitation_loss(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0]), 0).item() >= 0


def test_single_action_decision_never_produces_nan_loss() -> None:
    predicted = torch.tensor([0.5], requires_grad=True)

    loss = combined_imitation_loss(predicted, torch.tensor([3.0]), 0)

    assert torch.isfinite(loss)
