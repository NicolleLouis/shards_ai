from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

ADDITRI_GAIA_MANCienne = CardDefinition(
    "additri_gaia_mancienne", "Additri, Gaïamancienne", 5, Effect(),
    faction=Faction.MAQUIS, is_champion=True, champion_health=5,
    champion_ability=ChampionAbility(
        "gain_power_per_played_faction", amount=2, secondary_amount=2,
        faction=Faction.MAQUIS,
    ),
)

# Canonical uppercase export kept separate for readability in deck definitions.
ADDITRI_GAIAMENCIENNE = ADDITRI_GAIA_MANCienne
