from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

LEGIONNAIRE_KORVUS = CardDefinition(
    "legionnaire_korvus", "Légionnaire Korvus", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("gain_power", 2),
        Operation("recover_champion"),
    )),)),
    faction=Faction.HOMODEUS, shield=2,
)
