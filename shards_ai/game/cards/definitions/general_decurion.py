from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

GENERAL_DECURION = CardDefinition(
    "general_decurion", "Général Décurion", 7, Effect(),
    faction=Faction.HOMODEUS, is_champion=True, champion_health=7,
    champion_ability=ChampionAbility(
        "gain_gems_then_copy_faction", amount=3, threshold=20,
        faction=Faction.HOMODEUS,
    ),
)
