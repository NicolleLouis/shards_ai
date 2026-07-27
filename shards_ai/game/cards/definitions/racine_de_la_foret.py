from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

RACINE_DE_LA_FORET = CardDefinition(
    "racine_de_la_foret", "Racine de la Forêt", 7,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_health", 10), Operation("gain_power", 10, requires_union=True),
    )),)), faction=Faction.MAQUIS, is_mercenary=True,
)
