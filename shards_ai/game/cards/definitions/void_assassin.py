from ..model import CardDefinition, Effect
from ...enums import Faction

VOID_ASSASSIN = CardDefinition(
    card_id="void_assassin",
    name="Assassins du vide",
    cost=2,
    effect=Effect(power=5),
    faction=Faction.SPECTRA,
    is_mercenary=True,
)
