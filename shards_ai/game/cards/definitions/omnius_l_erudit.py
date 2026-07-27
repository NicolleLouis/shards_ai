from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

OMNIUS_L_ERUDIT = CardDefinition(
    "omnius_l_erudit", "Omnius l'érudit", 6,
    Effect(steps=(EffectStep(operations=(
        Operation("draw_card", 2), Operation("gain_mastery", 5, requires_domination=True),
    )),)), faction=Faction.ORDER, is_mercenary=True,
)
