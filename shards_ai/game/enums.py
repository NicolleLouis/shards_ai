from enum import Enum, IntEnum


class PlayerId(IntEnum):
    PLAYER_1 = 1
    PLAYER_2 = 2

    @property
    def opponent(self) -> "PlayerId":
        return PlayerId.PLAYER_2 if self is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


class Faction(str, Enum):
    NEUTRAL = "neutral"
    MAQUIS = "maquis"
    SPECTRA = "spectra"
    HOMODEUS = "homodeus"
    ORDER = "order"


class Phase(Enum):
    PLAY = "play"
    BUY = "buy"
    ATTACK = "attack"
    CLEANUP = "cleanup"


class GameStatus(Enum):
    RUNNING = "running"
    FINISHED = "finished"
    DRAW = "draw"
