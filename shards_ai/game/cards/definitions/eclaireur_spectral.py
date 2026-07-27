from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

ECLAIREUR_SPECTRAL = CardDefinition(
    "eclaireur_spectral", "Éclaireur spectral", 1,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 2),
        Operation("gain_power", 4, requires_echo=True),
    )),)), faction=Faction.SPECTRA,
)
