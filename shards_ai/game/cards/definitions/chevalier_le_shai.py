from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

CHEVALIER_LE_SHAI = CardDefinition(
    "chevalier_le_shai", "Chevalier Le'Shai", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 3), Operation("gain_power", 3, requires_union=True),
    )),)), faction=Faction.MAQUIS, is_mercenary=True,
)
