from ..random import GameRandom
from .definitions import HOMODEUS_CARDS, MAQUIS_CARDS, ORDER_CARDS, SPECTRA_CARDS
from .model import CardInstance

VOID_ASSASSIN_COUNT = 3
CENTRAL_DECK_SIZE = sum(
    count for _, count in (*MAQUIS_CARDS, *SPECTRA_CARDS, *ORDER_CARDS, *HOMODEUS_CARDS)
)


def build_central_deck(rng: GameRandom) -> list[CardInstance]:
    cards = []
    for definition, count in (
        *MAQUIS_CARDS, *SPECTRA_CARDS, *ORDER_CARDS, *HOMODEUS_CARDS
    ):
        cards.extend(
            CardInstance(f"central-{definition.card_id}-{index}", definition)
            for index in range(count)
        )
    rng.shuffle(cards)
    return cards
