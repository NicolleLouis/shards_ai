from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

MOINE_CRYPTOPOING = CardDefinition(
    "moine_cryptopoing", "Moine Cryptopoing", 5,
    Effect(steps=(EffectStep(operations=(Operation("draw_card"),)),)),
    faction=Faction.ORDER, shield=8,
)
