from ..model import CardDefinition, ChampionAbility, Effect, EffectStep, Operation
from ...enums import Faction

GIGA_ADEPTE_DE_LA_SOURCE = CardDefinition(
    "giga_adepte_de_la_source", "Giga, Adepte de la Source", 2, Effect(),
    faction=Faction.ORDER, is_champion=True, champion_health=4,
    on_play_effect=Effect(steps=(EffectStep(operations=(Operation("draw_card"),)),)),
    champion_ability=ChampionAbility("gain_mastery_if_domination", amount=3),
)
