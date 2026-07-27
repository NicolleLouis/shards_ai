from ..model import CardDefinition, ChampionAbility, Effect
from ...enums import Faction

IA_SYSTEMA = CardDefinition(
    "ia_systema", "I.A. Systema", 3, Effect(),
    faction=Faction.ORDER, is_champion=True, champion_health=4,
    champion_ability=ChampionAbility(
        "gain_mastery_then_draw", amount=1, threshold=20, draw_amount=2,
    ),
)
