from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

ZEN_CHI_SET = CardDefinition(
    "zen_chi_set", "Zen Chi Set, Fléau des dieux", 7, Effect(),
    faction=Faction.SPECTRA, is_champion=True, champion_health=5,
    champion_ability=ChampionAbility(
        "gain_power_then_recover_faction", amount=3, faction=Faction.SPECTRA,
    ),
)
