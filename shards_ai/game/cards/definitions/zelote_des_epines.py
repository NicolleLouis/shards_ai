from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

ZELOTE_DES_EPINES = CardDefinition(
    "zelote_des_epines", "Zélote des Épines", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("draw_card"),
        Operation("destroy_champion", requires_union=True),
    )),)),
    faction=Faction.MAQUIS, shield=3,
)
