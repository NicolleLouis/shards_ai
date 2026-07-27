from ..model import CardDefinition, Effect
from ...enums import Faction

CRYSTAL = CardDefinition(
    card_id="crystal",
    name="Cristal",
    cost=0,
    effect=Effect(gems=1),
    faction=Faction.NEUTRAL,
)
