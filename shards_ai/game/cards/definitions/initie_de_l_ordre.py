from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

INITIE_DE_L_ORDRE = CardDefinition(
    "initie_de_l_ordre", "Initié de l'Ordre", 1,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_gems", 2),
        Operation("gain_mastery", 2, requires_domination=True),
    )),)), faction=Faction.ORDER,
)
