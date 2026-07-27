from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

DRAKONARIUS = CardDefinition(
    "drakonarius", "Drakonarius", 6, Effect(),
    faction=Faction.HOMODEUS, is_champion=True, champion_health=2,
    champion_ability=ChampionAbility("gain_power", amount=6),
    passive_kind="drakonarius_protection",
)
