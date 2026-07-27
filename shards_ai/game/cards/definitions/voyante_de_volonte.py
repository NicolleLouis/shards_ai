from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

VOYANTE_DE_VOLONTE = CardDefinition(
    "voyante_de_volonte", "Voyante de Volonté", 4,
    Effect(steps=(EffectStep(operations=(Operation("gain_gems", 2),)),)),
    faction=Faction.ORDER, shield=5,
)
