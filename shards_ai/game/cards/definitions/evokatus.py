from ..model import CardDefinition, ChampionAbility, Effect, EffectStep, Operation
from ...enums import Faction

EVOKATUS = CardDefinition(
    "evokatus", "Evokatus", 4, Effect(),
    faction=Faction.HOMODEUS, is_champion=True, champion_health=2,
    on_play_effect=Effect(steps=(EffectStep(operations=(Operation("draw_card"),)),)),
    champion_ability=ChampionAbility(
        "gain_power_per_champion", amount=1, faction=Faction.HOMODEUS,
    ),
)
