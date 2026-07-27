from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

CLERC_AUX_SPORES = CardDefinition(
    "clerc_aux_spores", "Clerc aux Spores", 2,
    Effect(steps=(EffectStep(operations=(Operation("gain_health", 4),)),)),
    faction=Faction.MAQUIS,
    is_mercenary=True,
)
