from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

LE_GRAND_ARCHITECTE = CardDefinition(
    "le_grand_architecte", "Le Grand Architecte", 7,
    Effect(steps=(EffectStep(operations=(Operation("gain_mastery", 5),)),)),
    faction=Faction.ORDER,
    is_mercenary=True,
)
