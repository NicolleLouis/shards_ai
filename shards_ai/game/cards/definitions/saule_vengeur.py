from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

SAULE_VENGEUR = CardDefinition(
    "saule_vengeur", "Saule Vengeur", 4,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 4),
    )), EffectStep(operations=(
        Operation("gain_power", 4),
        Operation("destroy_all_champions"),
    ), mastery_at_least=15))),
    faction=Faction.MAQUIS,
    is_mercenary=True,
)
