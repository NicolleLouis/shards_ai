from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

DRONE_KILN = CardDefinition(
    "drone_kiln", "Drone Kiln", 1,
    Effect(steps=(EffectStep(operations=(Operation("gain_gems", 2),)),)),
    faction=Faction.HOMODEUS,
)
