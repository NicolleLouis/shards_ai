from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction


INFINITY_SHARD = CardDefinition(
    card_id="infinity_shard",
    name="Éclat de l'infini",
    cost=0,
    effect=Effect(
        steps=(
            EffectStep(
                mastery_at_least=30,
                operations=(Operation("win"),),
            ),
            EffectStep(
                mastery_at_least=20,
                operations=(Operation("gain_power", 5),),
            ),
            EffectStep(
                mastery_at_least=10,
                operations=(Operation("gain_power", 3),),
            ),
            EffectStep(operations=(Operation("gain_power", 2),)),
        )
    ),
    faction=Faction.NEUTRAL,
)
