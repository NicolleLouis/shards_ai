from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

PROPHETE_DE_LECLAT = CardDefinition(
    "prophete_de_leclat", "Prophète de l'éclat", 3,
    Effect(steps=(EffectStep(operations=(Operation("gain_mastery", 2),)),)),
    faction=Faction.ORDER,
    is_mercenary=True,
)
