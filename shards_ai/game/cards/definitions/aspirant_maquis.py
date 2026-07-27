from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

ASPIRANT_MAQUIS = CardDefinition(
    "aspirant_maquis", "Aspirant Maquis", 1,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_health", 3), Operation("gain_power", 5, requires_union=True),
    )),)), faction=Faction.MAQUIS,
)
