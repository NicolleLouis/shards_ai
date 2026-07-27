from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

OJAS = CardDefinition(
    "ojas", "Ojas, druide de la genèse", 4,
    Effect(steps=(
        EffectStep(mastery_at_least=20, operations=(Operation("copy_effect", 2),)),
        EffectStep(operations=(Operation("copy_effect", 1),)),
    )), faction=Faction.MAQUIS,
)
