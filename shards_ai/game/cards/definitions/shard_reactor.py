from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction


SHARD_REACTOR = CardDefinition(
    card_id="shard_reactor",
    name="Réacteur d'éclat",
    cost=0,
    effect=Effect(
        steps=(
            EffectStep(
                mastery_at_least=15,
                operations=(Operation("gain_gems", 4),),
            ),
            EffectStep(
                mastery_at_least=5,
                operations=(Operation("gain_gems", 3),),
            ),
            EffectStep(operations=(Operation("gain_gems", 2),)),
        )
    ),
    faction=Faction.NEUTRAL,
)
