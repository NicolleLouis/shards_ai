from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

VALKYRIE_DES_LANDES = CardDefinition(
    "valkyrie_des_landes", "Valkyrie des Landes", 4,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 4),
        Operation("lose_mastery", 2, requires_inspiration=True),
    )),)),
    faction=Faction.HOMODEUS,
    is_mercenary=True,
)
