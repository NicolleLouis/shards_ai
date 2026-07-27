from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

BROYEU_OPTIO = CardDefinition(
    "broyeu_optio", "Broyeur Optio", 5, Effect(),
    faction=Faction.HOMODEUS, is_champion=True, champion_health=4,
    champion_ability=ChampionAbility(
        "gain_power_threshold", amount=3, threshold=10, secondary_amount=2,
    ),
)
