from ..random import GameRandom
from .definitions import BLASTER, CRYSTAL, INFINITY_SHARD, SHARD_REACTOR
from .model import CardInstance

STARTER_DECK_SIZE = 10


def build_starter_deck(player_id: int, rng: GameRandom) -> list[CardInstance]:
    cards = [
        CardInstance(f"p{player_id}-crystal-{index}", CRYSTAL)
        for index in range(7)
    ]
    cards.append(CardInstance(f"p{player_id}-blaster-0", BLASTER))
    cards.append(CardInstance(f"p{player_id}-shard-reactor-0", SHARD_REACTOR))
    cards.append(CardInstance(f"p{player_id}-infinity-shard-0", INFINITY_SHARD))
    rng.shuffle(cards)
    return cards
