from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

BRISE_ETHER = CardDefinition(
    "brise_ether", "Brise-Éther", 4,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 4), Operation("gain_power", 4, mastery_at_least=10),
    )),)), faction=Faction.SPECTRA, is_mercenary=True,
)
