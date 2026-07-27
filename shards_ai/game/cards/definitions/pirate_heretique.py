from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

PIRATE_HERETIQUE = CardDefinition(
    "pirate_heretique", "Pirate Hérétique", 3,
    Effect(steps=(EffectStep(operations=(Operation("draw_card", 2),)),)),
    faction=Faction.ORDER,
    is_mercenary=True,
)
