from __future__ import annotations

import json

import pytest

from shards_ai.ai import (
    CARD_REPRESENTATION_SCHEMA_VERSION,
    CardSemanticRepresentation,
    clear_representation_cache,
    representation_for_definition,
)
from shards_ai.game import CARD_CATALOG, CardDefinition, Effect, EffectStep, Operation
from shards_ai.game.cards import ChampionAbility


def test_every_catalog_card_has_a_deterministic_semantic_representation() -> None:
    for definition in CARD_CATALOG.values():
        first = representation_for_definition(definition)
        second = representation_for_definition(definition)

        assert isinstance(first, CardSemanticRepresentation)
        assert first == second
        assert first.schema_version == CARD_REPRESENTATION_SCHEMA_VERSION
        assert first.card_definition_id == definition.card_id
        assert first.to_dict() == second.to_dict()


def test_representation_keeps_identity_and_structured_effects_for_drakonarius() -> None:
    representation = representation_for_definition(CARD_CATALOG["drakonarius"])

    assert representation.card_definition_id == "drakonarius"
    assert representation.is_champion is True
    assert representation.champion_health == 2
    assert representation.champion_ability is not None
    assert representation.champion_ability.kind == "gain_power"
    assert representation.passive_kind == "drakonarius_protection"


def test_representation_preserves_mastery_branches_and_operation_order() -> None:
    representation = representation_for_definition(CARD_CATALOG["infinity_shard"])

    assert [step.mastery_at_least for step in representation.effect.steps] == [30, 20, 10, None]
    assert [step.operations[0].kind for step in representation.effect.steps] == [
        "win",
        "gain_power",
        "gain_power",
        "gain_power",
    ]
    assert [step.operations[0].amount for step in representation.effect.steps] == [0, 5, 3, 2]


def test_representation_preserves_all_operation_constraints() -> None:
    definition = CardDefinition(
        card_id="structured-test",
        name="Not used by the representation",
        cost=4,
        faction=None,
        shield=2,
        effect=Effect(
            steps=(
                EffectStep(
                    operations=(
                        Operation(
                            "gain_power",
                            7,
                            target="self",
                            mastery_at_least=3,
                            health_at_least=40,
                            requires_union=True,
                            requires_echo=True,
                            requires_domination=True,
                            requires_inspiration=True,
                            recruit_to_hand_at_mastery=15,
                        ),
                    ),
                    mastery_at_least=10,
                ),
            ),
        ),
    )

    operation = representation_for_definition(definition).effect.steps[0].operations[0]

    assert operation.kind == "gain_power"
    assert operation.amount == 7
    assert operation.target == "self"
    assert operation.mastery_at_least == 3
    assert operation.health_at_least == 40
    assert operation.requires_union is True
    assert operation.requires_echo is True
    assert operation.requires_domination is True
    assert operation.requires_inspiration is True
    assert operation.recruit_to_hand_at_mastery == 15


def test_representation_is_json_serializable_and_excludes_human_name() -> None:
    representation = representation_for_definition(CARD_CATALOG["drakonarius"])

    serialized = json.dumps(representation.to_dict(), sort_keys=True)

    assert "drakonarius" in serialized
    assert "Drakonarius" not in serialized
    assert "schema_version" in serialized


def test_cache_can_be_cleared_without_changing_the_representation() -> None:
    definition = CARD_CATALOG["crystal"]
    first = representation_for_definition(definition)
    clear_representation_cache()
    second = representation_for_definition(definition)

    assert first == second
    assert first is not second


def test_unknown_operation_kind_fails_explicitly() -> None:
    definition = CardDefinition(
        card_id="unknown-operation",
        name="Unknown operation",
        cost=0,
        effect=Effect(steps=(EffectStep((Operation("future_operation"),)),)),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Unsupported operation kind"):
        representation_for_definition(definition)


def test_unknown_champion_ability_kind_fails_explicitly() -> None:
    definition = CardDefinition(
        card_id="unknown-ability",
        name="Unknown ability",
        cost=0,
        is_champion=True,
        champion_health=1,
        effect=Effect(),
        champion_ability=ChampionAbility("future_ability"),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Unsupported champion ability kind"):
        representation_for_definition(definition)
