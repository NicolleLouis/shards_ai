from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

PRIMUS_PILUS = CardDefinition(
    "primus_pilus", "Primus Pilus", 2, Effect(),
    faction=Faction.HOMODEUS, is_champion=True, champion_health=6,
    champion_ability=ChampionAbility(
        "draw_if_champion_faction_count", faction=Faction.HOMODEUS,
        threshold=3, draw_amount=2,
    ),
)
