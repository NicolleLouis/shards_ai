from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

ERMITE_FONGIQUE = CardDefinition(
    "ermite_fongique", "Ermite Fongique", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_mastery", 1), Operation("gain_health", 5, mastery_at_least=10),
    )),)), faction=Faction.MAQUIS, is_mercenary=True,
)
