from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

DRONES_NUMERI = CardDefinition(
    "drones_numeri", "Drones Numeri", 3, Effect(),
    faction=Faction.HOMODEUS, is_champion=True, champion_health=5,
    champion_ability=ChampionAbility("gain_gem_and_arm_recruitment", amount=1),
)
