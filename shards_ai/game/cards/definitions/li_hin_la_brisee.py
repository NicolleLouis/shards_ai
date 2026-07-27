from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

LI_HIN_LA_BRISEE = CardDefinition(
    "li_hin_la_brisee", "Li Hin, la Brisée", 3, Effect(),
    faction=Faction.SPECTRA, is_champion=True, champion_health=1,
    champion_ability=ChampionAbility("gain_power", amount=1),
    passive_kind="li_hin_immunity",
)
