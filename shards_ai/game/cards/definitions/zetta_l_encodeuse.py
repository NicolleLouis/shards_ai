from ..model import CardDefinition, Effect
from ...enums import Faction

ZETTA_L_ENCODEUSE = CardDefinition(
    "zetta_l_encodeuse", "Zetta, l'encodeuse", 5, Effect(),
    faction=Faction.ORDER, shield=5, is_champion=True, champion_health=5,
    passive_kind="zetta_protection",
)
