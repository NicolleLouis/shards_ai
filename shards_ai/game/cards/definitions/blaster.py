from ..model import CardDefinition, Effect
from ...enums import Faction

BLASTER = CardDefinition(
    card_id="blaster",
    name="Blaster",
    cost=0,
    effect=Effect(power=1),
    faction=Faction.NEUTRAL,
)
