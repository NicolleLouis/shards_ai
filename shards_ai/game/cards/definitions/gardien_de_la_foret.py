from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

GARDIEN_DE_LA_FORET = CardDefinition(
    "gardien_de_la_foret", "Gardien de la Forêt", 4,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 2), Operation("draw_card"),
        Operation("gain_health", 6, requires_union=True),
    )),)), faction=Faction.MAQUIS,
)
