from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

GARDE_MEMOIRE = CardDefinition(
    "garde_memoire", "Garde Mémoire", 2,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_mastery", 1), Operation("draw_card", mastery_at_least=10),
    )),)), faction=Faction.ORDER, is_mercenary=True,
)
