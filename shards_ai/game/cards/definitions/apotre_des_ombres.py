from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

APOTRE_DES_OMBRES = CardDefinition(
    "apotre_des_ombres", "Apôtre des ombres", 2,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 2), Operation("offer_banish", 1),
    )),)), faction=Faction.SPECTRA, is_mercenary=True,
)
