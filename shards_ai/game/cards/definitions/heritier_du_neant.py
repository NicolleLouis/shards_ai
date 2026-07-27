from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

HERITIER_DU_NEANT = CardDefinition(
    "heritier_du_neant", "Héritier du Néant", 5,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 3),
        Operation("gain_power_per_discard_faction", 2, requires_echo=True, faction=Faction.SPECTRA),
    )),)), faction=Faction.SPECTRA, is_mercenary=True,
)
