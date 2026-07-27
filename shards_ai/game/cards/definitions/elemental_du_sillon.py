from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

ELEMENTAL_DU_SILLON = CardDefinition(
    "elemental_du_sillon", "Élémental du Sillon", 5,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_health", 4), Operation("draw_card"),
        Operation("gain_power", 6, health_at_least=50),
    )),)), faction=Faction.MAQUIS,
)
