from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

DRONES_MINIERS = CardDefinition(
    "drones_miniers", "Drones Miniers", 2,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_gems", 1), Operation("draw_card"),
    )),)), faction=Faction.HOMODEUS,
)
