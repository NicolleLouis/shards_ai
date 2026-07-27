from dataclasses import dataclass, field

from .cards import CardInstance
from .enums import GameStatus, Phase, PlayerId


@dataclass(slots=True)
class PlayerState:
    player_id: PlayerId
    health: int = 50
    gems: int = 0
    mastery: int = 0
    mastery_action_used: bool = False
    power: int = 0
    hand: list[CardInstance] = field(default_factory=list)
    draw_pile: list[CardInstance] = field(default_factory=list)
    discard_pile: list[CardInstance] = field(default_factory=list)
    play_zone: list[CardInstance] = field(default_factory=list)
    champions: list[CardInstance] = field(default_factory=list)
    activated_champion_ids: set[str] = field(default_factory=set)
    played_card_ids_this_turn: set[str] = field(default_factory=set)
    recruited_mercenary_ids_this_turn: set[str] = field(default_factory=set)
    pending_decision: "PendingDecision | None" = None
    pending_homodeus_champion_recruitment: bool = False
    pending_banishes: int = 0
    pending_free_recruit_cost: int | None = None
    pending_free_recruit_to_hand: bool = False

    @property
    def pending_damage(self) -> int:
        """Deprecated compatibility name for the player's Power."""
        return self.power

    @pending_damage.setter
    def pending_damage(self, amount: int) -> None:
        self.power = amount

    @property
    def damage(self) -> int:
        """Deprecated compatibility view for the player's Power."""
        return self.power


@dataclass(slots=True)
class GameState:
    players: dict[PlayerId, PlayerState]
    active_player: PlayerId
    starting_player: PlayerId | None = None
    central_deck: list[CardInstance] = field(default_factory=list)
    river: list[CardInstance | None] = field(default_factory=list)
    phase: Phase = Phase.PLAY
    status: GameStatus = GameStatus.RUNNING
    winner: PlayerId | None = None
    turn_number: int = 1
    seed: int | None = None

    @property
    def pending_damage(self) -> int:
        """Deprecated compatibility view of the active player's Power."""
        return self.players[self.active_player].power

    @pending_damage.setter
    def pending_damage(self, amount: int) -> None:
        self.players[self.active_player].power = amount


@dataclass(frozen=True, slots=True)
class PendingDecision:
    kind: str
    candidates: tuple[str, ...]
