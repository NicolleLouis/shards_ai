from ..model import CardDefinition, Effect, EffectStep, Operation
from ...enums import Faction

MOINE_DU_PORTAIL = CardDefinition(
    "moine_du_portail", "Moine du portail", 3,
    Effect(steps=(EffectStep(operations=(
        Operation("recruit_free_card", 6, recruit_to_hand_at_mastery=15),
    )),)), faction=Faction.ORDER,
)
