from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

ZARA_RA = CardDefinition(
    "zara_ra", "Zara Ra, Écorcheur d’âme", 5,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 4), Operation("gain_mastery", 1),
        Operation("offer_banish", 2, mastery_at_least=10),
    )),)), faction=Faction.SPECTRA, is_mercenary=True,
)
