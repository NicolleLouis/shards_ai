from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

FLEAU_DES_OMBRES = CardDefinition(
    "fleau_des_ombres", "Fléau des Ombres", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_mastery", 1), Operation("offer_banish", 1),
    )),)), faction=Faction.SPECTRA, is_mercenary=True,
)
