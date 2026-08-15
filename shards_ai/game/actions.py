from dataclasses import dataclass


class Action:
    """Marker base class for public game actions."""


@dataclass(frozen=True, slots=True)
class PlayCard(Action):
    card_id: str


@dataclass(frozen=True, slots=True)
class BanishCard(Action):
    card_id: str


@dataclass(frozen=True, slots=True)
class SkipBanish(Action):
    pass


@dataclass(frozen=True, slots=True)
class RecruitFreeCard(Action):
    river_slot: int
    card_instance_id: str


@dataclass(frozen=True, slots=True)
class PassPlayPhase(Action):
    pass


@dataclass(frozen=True, slots=True)
class GainMastery(Action):
    pass


@dataclass(frozen=True, slots=True)
class BuyCard(Action):
    river_slot: int
    card_instance_id: str


@dataclass(frozen=True, slots=True)
class RecruitMercenary(Action):
    river_slot: int
    card_instance_id: str


@dataclass(frozen=True, slots=True)
class StopBuying(Action):
    pass


@dataclass(frozen=True, slots=True)
class EndMainPhase(Action):
    """End the modern interleaved main phase and start the attack phase."""

    pass


@dataclass(frozen=True, slots=True)
class AssignPower(Action):
    amount: int
    target: str = "opponent"


@dataclass(frozen=True, slots=True)
class ActivateChampion(Action):
    champion_id: str


@dataclass(frozen=True, slots=True)
class ChoosePendingDecision(Action):
    choice_id: str


# Deprecated compatibility alias. Power is the resource assigned during ATTACK.
AssignDamage = AssignPower
