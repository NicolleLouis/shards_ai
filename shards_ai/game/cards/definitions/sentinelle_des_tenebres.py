from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

SENTINELLE_DES_TENEBRES = CardDefinition(
    "sentinelle_des_tenebres", "Sentinelle des ténèbres", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 3), Operation("recover_mercenary"),
    )),)),
    faction=Faction.SPECTRA,
)
