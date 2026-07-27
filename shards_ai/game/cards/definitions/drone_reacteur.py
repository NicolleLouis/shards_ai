from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

DRONE_REACTEUR = CardDefinition(
    "drone_reacteur", "Drone Réacteur", 3,
    Effect(steps=(EffectStep(operations=(Operation("gain_gems", 3),)),)),
    faction=Faction.HOMODEUS,
)
