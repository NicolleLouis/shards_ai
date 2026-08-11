from __future__ import annotations

from .actions import (
    Action,
    ActivateChampion,
    AssignPower,
    BanishCard,
    BuyCard,
    ChoosePendingDecision,
    GainMastery,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
    StopBuying,
    SkipBanish,
)
from .cards import (
    CardDefinition,
    CardInstance,
    build_central_deck,
    build_starter_deck,
)
from .enums import Faction, GameStatus, Phase, PlayerId
from .errors import InvalidActionError, InvalidGameStateError
from .random import GameRandom
from .observation import NeuralObservation, build_neural_observation
from .state import GameState, PlayerState
from .state import PendingDecision


class Game:
    """Mutable, deterministic duel engine with a central deck and buying phase."""

    STARTING_HAND_SIZE = 5
    STARTING_HEALTH = 50
    RIVER_SIZE = 6
    MIN_MASTERY = 0
    MAX_MASTERY = 30
    MASTERY_COST = 1
    MASTERY_GAIN = 1

    def __init__(self, state: GameState, rng: GameRandom) -> None:
        self.state = state
        self._rng = rng

    @classmethod
    def new(
        cls,
        seed: int | None = None,
        card_definition: CardDefinition | None = None,
        rng: GameRandom | None = None,
    ) -> "Game":
        rng = rng or GameRandom(seed)
        players: dict[PlayerId, PlayerState] = {}

        for player_id in PlayerId:
            if card_definition is None:
                deck = build_starter_deck(player_id.value, rng)
            else:
                deck = [
                    CardInstance(
                        instance_id=f"p{player_id.value}-card-{index}",
                        definition=card_definition,
                    )
                    for index in range(10)
                ]
                rng.shuffle(deck)
            players[player_id] = PlayerState(
                player_id=player_id,
                health=cls.STARTING_HEALTH,
                draw_pile=deck,
            )

        central_deck = build_central_deck(rng)
        river: list[CardInstance | None] = [None] * cls.RIVER_SIZE
        for slot in range(cls.RIVER_SIZE):
            river[slot] = central_deck.pop()

        starting_player = rng.choice(list(PlayerId))
        players[starting_player].mastery = 0
        players[starting_player.opponent].mastery = 1

        game = cls(
            state=GameState(
                players=players,
                active_player=starting_player,
                starting_player=starting_player,
                central_deck=central_deck,
                river=river,
                seed=seed,
            ),
            rng=rng,
        )
        for player in players.values():
            game.draw_many(player.player_id, cls.STARTING_HAND_SIZE)
        return game

    def observation_for(self, player_id: PlayerId) -> GameState:
        """Return a detached observation; hidden information is not modelled yet."""
        if player_id not in self.state.players:
            raise InvalidActionError(f"Unknown player: {player_id}")
        return self._detached_state()

    def neural_observation_for(self, player_id: PlayerId) -> NeuralObservation:
        """Return the information-masked observation for the active neural player."""
        if player_id not in self.state.players:
            raise InvalidActionError(f"Unknown player: {player_id}")
        if player_id is not self.state.active_player:
            raise InvalidActionError("Neural observations are only available for the active player")
        return build_neural_observation(self.state)

    def shaping_observation_for(self, player_id: PlayerId) -> GameState:
        """Return the detached state needed by transition reward shaping."""
        if player_id not in self.state.players:
            raise InvalidActionError(f"Unknown player: {player_id}")
        state = self.state
        players = {}
        for current_id, player in state.players.items():
            if current_id is player_id:
                players[current_id] = PlayerState(
                    player_id=player.player_id,
                    hand=list(player.hand),
                    draw_pile=list(player.draw_pile),
                    discard_pile=list(player.discard_pile),
                    play_zone=list(player.play_zone),
                    champions=list(player.champions),
                )
            else:
                players[current_id] = PlayerState(
                    player_id=player.player_id,
                    champions=list(player.champions),
                )
        return GameState(
            players=players,
            active_player=state.active_player,
            starting_player=state.starting_player,
            phase=state.phase,
            status=state.status,
            winner=state.winner,
            turn_number=state.turn_number,
            seed=state.seed,
        )

    def _detached_state(self) -> GameState:
        """Copy mutable game data while sharing immutable card definitions."""
        state = self.state
        copy_card = self._copy_card
        players = {
            player_id: PlayerState(
                player_id=player.player_id,
                health=player.health,
                gems=player.gems,
                mastery=player.mastery,
                mastery_action_used=player.mastery_action_used,
                power=player.power,
                hand=[copy_card(card) for card in player.hand],
                draw_pile=[copy_card(card) for card in player.draw_pile],
                discard_pile=[copy_card(card) for card in player.discard_pile],
                play_zone=[copy_card(card) for card in player.play_zone],
                champions=[copy_card(card) for card in player.champions],
                activated_champion_ids=set(player.activated_champion_ids),
                played_card_ids_this_turn=set(player.played_card_ids_this_turn),
                recruited_mercenary_ids_this_turn=set(player.recruited_mercenary_ids_this_turn),
                pending_decision=player.pending_decision,
                pending_homodeus_champion_recruitment=player.pending_homodeus_champion_recruitment,
                pending_banishes=player.pending_banishes,
                pending_free_recruit_cost=player.pending_free_recruit_cost,
                pending_free_recruit_to_hand=player.pending_free_recruit_to_hand,
            )
            for player_id, player in state.players.items()
        }
        return GameState(
            players=players,
            active_player=state.active_player,
            starting_player=state.starting_player,
            central_deck=[copy_card(card) for card in state.central_deck],
            river=[copy_card(card) if card is not None else None for card in state.river],
            phase=state.phase,
            status=state.status,
            winner=state.winner,
            turn_number=state.turn_number,
            seed=state.seed,
        )

    @staticmethod
    def _copy_card(card: CardInstance) -> CardInstance:
        copied = CardInstance.__new__(CardInstance)
        copied.instance_id = card.instance_id
        copied.definition = card.definition
        return copied

    def clone(self) -> "Game":
        """Return a detached copy suitable for tests, replay and search."""
        # ``GameState`` already has an explicit detached copier that shares immutable
        # card definitions.  Copying the entire object graph with ``deepcopy`` is
        # disproportionately expensive for the bounded PLAY solver, which clones the
        # game once per explored branch.  The RNG remains a private mutable stream and
        # must retain its exact state in the clone.
        return Game(state=self._detached_state(), rng=self._rng.clone())

    def legal_actions(self) -> list[Action]:
        if self.state.status is not GameStatus.RUNNING:
            return []
        if self.active.pending_decision is not None:
            return [ChoosePendingDecision(choice) for choice in self.active.pending_decision.candidates]
        if self.active.pending_free_recruit_cost is not None:
            return [
                RecruitFreeCard(slot, card.instance_id)
                for slot, card in enumerate(self.state.river)
                if card is not None
                and card.definition.cost <= self.active.pending_free_recruit_cost
            ]
        if self.active.pending_banishes:
            return [
                *[BanishCard(card.instance_id) for card in self._banishable_cards(self.active)],
                SkipBanish(),
            ]
        if self.state.phase is Phase.PLAY:
            hand_actions = [PlayCard(card.instance_id) for card in self.active.hand]
            champion_actions = [
                ActivateChampion(card.instance_id)
                for card in self.active.champions
                if card.definition.champion_ability is not None
                and card.instance_id not in self.active.activated_champion_ids
            ]
            mastery_actions = []
            if self._can_gain_mastery():
                mastery_actions.append(GainMastery())
            return [*hand_actions, *champion_actions, *mastery_actions, PassPlayPhase()]
        if self.state.phase is Phase.BUY:
            buy_actions = [
                BuyCard(slot, card.instance_id)
                for slot, card in enumerate(self.state.river)
                if card is not None and card.definition.cost <= self.active.gems
            ]
            recruit_actions = [
                RecruitMercenary(slot, card.instance_id)
                for slot, card in enumerate(self.state.river)
                if card is not None
                and card.definition.is_mercenary
                and card.definition.cost <= self.active.gems
            ]
            return [*buy_actions, *recruit_actions, StopBuying()]
        if self.state.phase is Phase.ATTACK:
            targets = ["opponent"]
            targets.extend(self._legal_attack_champion_ids())
            return [AssignPower(self.active.power, target=target) for target in targets]
        return []

    def apply(self, action: Action) -> None:
        if self.state.status is not GameStatus.RUNNING:
            raise InvalidActionError("The game is already finished")
        active = self.state.players[self.state.active_player]
        if active.pending_decision is not None and not isinstance(action, ChoosePendingDecision):
            raise InvalidActionError("A pending decision must be resolved first")
        if active.pending_free_recruit_cost is not None and not isinstance(action, RecruitFreeCard):
            raise InvalidActionError("A free recruitment decision is pending")
        if active.pending_banishes and not isinstance(action, (BanishCard, SkipBanish)):
            raise InvalidActionError("A banishment decision is pending")
        action_type = type(action)
        if action_type is ChoosePendingDecision:
            self._choose_pending_decision(action)
        elif action_type is ActivateChampion:
            self._activate_champion(action)
        elif action_type is PlayCard:
            self._play_card(action)
        elif action_type is BanishCard:
            self._banish_card(action)
        elif action_type is SkipBanish:
            self._skip_banish()
        elif action_type is RecruitFreeCard:
            self._recruit_free_card(action)
        elif action_type is GainMastery:
            self._gain_mastery()
        elif action_type is PassPlayPhase:
            self._pass_play_phase()
        elif action_type is BuyCard:
            self._buy_card(action)
        elif action_type is RecruitMercenary:
            self._recruit_mercenary(action)
        elif action_type is StopBuying:
            self._stop_buying()
        elif action_type is AssignPower:
            self._assign_power(action)
        elif isinstance(action, PlayCard):
            self._play_card(action)
        elif isinstance(action, BanishCard):
            self._banish_card(action)
        elif isinstance(action, SkipBanish):
            self._skip_banish()
        elif isinstance(action, RecruitFreeCard):
            self._recruit_free_card(action)
        elif isinstance(action, GainMastery):
            self._gain_mastery()
        elif isinstance(action, PassPlayPhase):
            self._pass_play_phase()
        elif isinstance(action, BuyCard):
            self._buy_card(action)
        elif isinstance(action, RecruitMercenary):
            self._recruit_mercenary(action)
        elif isinstance(action, StopBuying):
            self._stop_buying()
        elif isinstance(action, AssignPower):
            self._assign_power(action)
        else:
            raise InvalidActionError(f"Unsupported action: {action!r}")

    @property
    def active(self) -> PlayerState:
        return self.state.players[self.state.active_player]

    @property
    def opponent(self) -> PlayerState:
        return self.state.players[self.state.active_player.opponent]

    def draw_one(self, player_id: PlayerId) -> CardInstance | None:
        drawn = self.draw_many(player_id, 1)
        return drawn[0] if drawn else None

    def draw_many(self, player_id: PlayerId, amount: int) -> list[CardInstance]:
        if amount < 0:
            raise ValueError("Draw amount cannot be negative")
        player = self.state.players[player_id]
        drawn: list[CardInstance] = []

        while amount:
            # A draw is a request for up to ``amount`` cards. Banishes can
            # permanently reduce a deck below the usual hand size, so an
            # exhausted draw pile and discard pile simply end the draw.
            if not player.draw_pile and not player.discard_pile:
                break
            draw_pile = self._ensure_draw_pile(player)
            draw_count = min(amount, len(draw_pile))
            cards = draw_pile[-draw_count:]
            del draw_pile[-draw_count:]
            cards.reverse()
            player.hand.extend(cards)
            drawn.extend(cards)
            amount -= draw_count
        return drawn

    def _ensure_draw_pile(self, player: PlayerState) -> list[CardInstance]:
        if player.draw_pile:
            return player.draw_pile
        if not player.discard_pile:
            raise InvalidGameStateError(
                f"Player {player.player_id} has no card available to draw"
            )
        self._rng.shuffle(player.discard_pile)
        player.draw_pile.extend(player.discard_pile)
        player.discard_pile.clear()
        return player.draw_pile

    def _play_card(self, action: PlayCard) -> None:
        if self.state.phase is not Phase.PLAY:
            raise InvalidActionError(
                f"Action requires phase {Phase.PLAY.value}, got {self.state.phase.value}"
            )
        player = self.active
        for card_index, candidate in enumerate(player.hand):
            if candidate.instance_id == action.card_id:
                card = player.hand.pop(card_index)
                break
        else:
            raise InvalidActionError(
                f"Card is not in the active player's hand: {action.card_id}"
            )

        player.played_card_ids_this_turn.add(card.instance_id)
        if card.definition.is_champion:
            player.champions.append(card)
            if card.definition.on_play_effect is not None:
                self._resolve_effect(player, card, card.definition.on_play_effect)
        else:
            player.play_zone.append(card)
            self._resolve_card_effect(player, card)

    def _activate_champion(self, action: ActivateChampion, *, immediate: bool = False) -> None:
        player = self.active
        if not immediate and self.state.phase is not Phase.PLAY:
            raise InvalidActionError("Champion activation requires the play phase")
        if action.champion_id in player.activated_champion_ids:
            raise InvalidActionError("This champion has already been activated this turn")
        champion = next(
            (card for card in player.champions if card.instance_id == action.champion_id),
            None,
        )
        if champion is None or champion.definition.champion_ability is None:
            raise InvalidActionError("The selected champion has no activatable ability")
        self._resolve_champion_ability(player, champion)
        player.activated_champion_ids.add(champion.instance_id)

    def _resolve_champion_ability(self, player: PlayerState, champion: CardInstance) -> None:
        ability = champion.definition.champion_ability
        if ability is None:
            return
        kind = ability.kind
        if kind == "gain_power":
            player.power += ability.amount
        elif kind == "gain_power_per_played_faction":
            player.power += ability.amount + ability.secondary_amount * sum(
                self._card_by_id(player, card_id).definition.faction is ability.faction
                for card_id in player.played_card_ids_this_turn
                if self._card_by_id(player, card_id) is not None
            )
        elif kind == "gain_mastery_then_draw":
            player.mastery = self._clamp_mastery(player.mastery + ability.amount)
            if ability.threshold is not None and player.mastery >= ability.threshold:
                self.draw_many(player.player_id, ability.draw_amount)
        elif kind == "draw_if_domination":
            if self._has_domination(player, champion):
                self.draw_many(player.player_id, ability.draw_amount)
                player.mastery = self._clamp_mastery(player.mastery + ability.amount)
        elif kind == "gain_mastery_if_domination":
            if self._has_domination(player, champion):
                player.mastery = self._clamp_mastery(player.mastery + ability.amount)
        elif kind == "draw_if_champion_faction_count":
            if sum(card.definition.faction is ability.faction for card in player.champions) >= (ability.threshold or 0):
                self.draw_many(player.player_id, ability.draw_amount)
        elif kind == "gain_gem_and_arm_recruitment":
            player.gems += ability.amount
            player.pending_homodeus_champion_recruitment = True
        elif kind == "gain_health_per_champion":
            player.health = min(
                self.STARTING_HEALTH,
                player.health + ability.amount * sum(
                    card.definition.faction is ability.faction for card in player.champions
                ),
            )
        elif kind == "gain_power_per_champion":
            player.power += ability.amount * sum(
                card.definition.faction is ability.faction for card in player.champions
            )
        elif kind == "gain_power_threshold":
            player.power += ability.amount
            if ability.threshold is not None and player.mastery >= ability.threshold:
                player.power += ability.secondary_amount
        elif kind == "gain_power_then_recover_faction":
            player.power += ability.amount
            self._offer_decision(
                player,
                "select_faction_discard",
                [
                    card.instance_id
                    for card in player.discard_pile
                    if card.definition.faction is ability.faction
                ],
            )
        elif kind == "gain_gems_then_copy_faction":
            player.gems += ability.amount
            if ability.threshold is None or player.mastery >= ability.threshold:
                self._offer_decision(
                    player,
                    "select_effect_copy",
                    [
                        card_id
                        for card_id in player.played_card_ids_this_turn
                        if (card := self._card_by_id(player, card_id)) is not None
                        and card.definition.faction is ability.faction
                        and not card.definition.is_champion
                    ],
                )
        else:
            raise InvalidActionError(f"Unsupported champion ability: {kind}")

    def _card_by_id(self, player: PlayerState, instance_id: str) -> CardInstance | None:
        return next(
            (
                card
                for zone in (player.hand, player.play_zone, player.champions, player.discard_pile)
                for card in zone
                if card.instance_id == instance_id
            ),
            None,
        )

    def _offer_decision(self, player: PlayerState, kind: str, candidates: list[str]) -> None:
        if not candidates:
            return
        if len(candidates) == 1 and kind not in {"destroy_all_champions", "select_effect_copy"}:
            self._resolve_decision_choice(player, kind, candidates[0], tuple(candidates))
            return
        player.pending_decision = PendingDecision(kind, tuple(candidates))

    def _choose_pending_decision(self, action: ChoosePendingDecision) -> None:
        player = self.active
        decision = player.pending_decision
        if decision is None or action.choice_id not in decision.candidates:
            raise InvalidActionError("The selected pending decision is not legal")
        self._resolve_decision_choice(player, decision.kind, action.choice_id, decision.candidates)

    def _resolve_decision_choice(
        self, player: PlayerState, kind: str, choice_id: str, candidates: tuple[str, ...]
    ) -> None:
        player.pending_decision = None
        if kind in {"select_spectra_discard", "select_faction_discard", "select_champion_discard"}:
            for index, card in enumerate(player.discard_pile):
                if card.instance_id == choice_id:
                    player.hand.append(player.discard_pile.pop(index))
                    return
            raise InvalidActionError("The selected card is no longer in the discard pile")
        if kind == "select_mercenary_discard":
            for index, card in enumerate(player.discard_pile):
                if card.instance_id == choice_id and card.definition.is_mercenary:
                    player.hand.append(player.discard_pile.pop(index))
                    return
            raise InvalidActionError("The selected mercenary is no longer in the discard pile")
        if kind in {"destroy_opponent_champion", "destroy_all_champions"}:
            self._destroy_champion(self.opponent, choice_id)
            remaining = tuple(candidate for candidate in candidates if candidate != choice_id)
            if kind == "destroy_all_champions" and remaining:
                player.pending_decision = PendingDecision(kind, remaining)
            return
        if kind == "select_effect_copy":
            card = self._card_by_id(player, choice_id)
            if card is None or card.definition.is_champion:
                raise InvalidActionError("The selected effect is no longer available")
            self._resolve_card_effect(player, card)
            remaining = tuple(candidate for candidate in candidates if candidate != choice_id)
            if remaining:
                player.pending_decision = PendingDecision(kind, remaining)
            return
        raise InvalidActionError(f"Unsupported pending decision: {kind}")

    def _destroy_champion(self, owner: PlayerState, champion_id: str) -> None:
        for index, card in enumerate(owner.champions):
            if card.instance_id == champion_id:
                owner.discard_pile.append(owner.champions.pop(index))
                return
        raise InvalidActionError("The selected champion is no longer on the board")

    def _legal_attack_champion_ids(self) -> list[str]:
        opponent = self.opponent
        protected_by_zetta = any(
            card.definition.passive_kind == "zetta_protection" for card in opponent.champions
        )
        has_general = any(
            card.definition.card_id == "general_decurion" for card in opponent.champions
        )
        return [
            card.instance_id
            for card in opponent.champions
            if self.active.power >= (card.definition.champion_health or 0)
            and not card.definition.passive_kind == "li_hin_immunity"
            and not (protected_by_zetta and card.definition.passive_kind != "zetta_protection")
            and not (card.definition.passive_kind == "drakonarius_protection" and has_general)
        ]

    def _resolve_card_effect(self, player: PlayerState, card: CardInstance) -> None:
        self._resolve_effect(player, card, card.definition.effect)

    def _resolve_effect(self, player: PlayerState, card: CardInstance, effect) -> None:
        if not effect.steps:
            player.gems += effect.gems
            player.power += effect.power
            return

        opponent = self.state.players[player.player_id.opponent]
        for operation in effect.operations_for_mastery(player.mastery):
            if (
                operation.mastery_at_least is not None
                and player.mastery < operation.mastery_at_least
            ):
                continue
            if operation.health_at_least is not None and player.health < operation.health_at_least:
                continue
            if operation.requires_union and not self._has_union_card(player, card):
                continue
            if operation.requires_echo and not self._has_echo_card(player):
                continue
            if operation.requires_domination and not self._has_domination(player, card):
                continue
            if operation.requires_inspiration and not player.champions:
                continue
            if operation.kind == "gain_gems":
                player.gems += operation.amount
            elif operation.kind == "gain_power":
                player.power += operation.amount
            elif operation.kind == "gain_health":
                player.health = min(self.STARTING_HEALTH, player.health + operation.amount)
            elif operation.kind == "gain_mastery":
                player.mastery = self._clamp_mastery(player.mastery + operation.amount)
            elif operation.kind == "lose_mastery":
                target = player if operation.target == "self" else opponent
                target.mastery = self._clamp_mastery(target.mastery - operation.amount)
            elif operation.kind == "draw_card":
                self.draw_many(player.player_id, max(1, operation.amount))
            elif operation.kind == "copy_effect":
                candidates = [candidate for candidate in player.play_zone[:-1] if not candidate.definition.is_champion]
                if candidates:
                    for _ in range(operation.amount):
                        self._resolve_card_effect(player, candidates[-1])
            elif operation.kind == "offer_banish":
                player.pending_banishes += operation.amount
            elif operation.kind == "recruit_free_card":
                self._offer_free_recruit(
                    player,
                    operation.amount,
                    operation.recruit_to_hand_at_mastery is not None
                    and player.mastery >= operation.recruit_to_hand_at_mastery,
                )
            elif operation.kind == "destroy_champion":
                candidates = [card.instance_id for card in opponent.champions]
                self._offer_decision(player, "destroy_opponent_champion", candidates)
            elif operation.kind == "destroy_all_champions":
                candidates = [card.instance_id for card in opponent.champions]
                self._offer_decision(player, "destroy_all_champions", candidates)
            elif operation.kind == "recover_champion":
                candidates = [
                    card.instance_id
                    for card in player.discard_pile
                    if card.definition.is_champion
                ]
                self._offer_decision(player, "select_champion_discard", candidates)
            elif operation.kind == "recover_mercenary":
                candidates = [
                    card.instance_id
                    for card in player.discard_pile
                    if card.definition.is_mercenary
                ]
                self._offer_decision(player, "select_mercenary_discard", candidates)
            elif operation.kind == "gain_power_per_discard_faction":
                player.power += operation.amount * sum(
                    candidate.definition.faction is operation.faction
                    for candidate in player.discard_pile
                )
            elif operation.kind == "deal_damage":
                if operation.target != "opponent":
                    raise InvalidActionError(
                        f"Unsupported damage target: {operation.target}"
                    )
                opponent.health -= operation.amount
                if opponent.health <= 0:
                    self.state.status = GameStatus.FINISHED
                    self.state.winner = player.player_id
                    return
            elif operation.kind == "win":
                self.state.status = GameStatus.FINISHED
                self.state.winner = player.player_id
                return
            else:
                raise InvalidActionError(f"Unsupported card operation: {operation.kind}")

    def _can_gain_mastery(self) -> bool:
        return (
            self.active.gems >= self.MASTERY_COST
            and self.active.mastery < self.MAX_MASTERY
            and not self.active.mastery_action_used
        )

    def _gain_mastery(self) -> None:
        if self.state.phase is not Phase.PLAY:
            raise InvalidActionError(
                f"Action requires phase {Phase.PLAY.value}, got {self.state.phase.value}"
            )
        if not self._can_gain_mastery():
            raise InvalidActionError("GainMastery is not currently legal")
        self.active.gems -= self.MASTERY_COST
        self.active.mastery = self._clamp_mastery(
            self.active.mastery + self.MASTERY_GAIN
        )
        self.active.mastery_action_used = True

    def _clamp_mastery(self, value: int) -> int:
        return max(self.MIN_MASTERY, min(self.MAX_MASTERY, value))

    def _has_union_card(self, player: PlayerState, current: CardInstance) -> bool:
        return any(
            candidate is not current
            and candidate.definition.faction is not None
            and candidate.definition.faction is current.definition.faction
            for candidate in [*player.hand, *player.play_zone]
        )

    def _has_echo_card(self, player: PlayerState) -> bool:
        return any(card.definition.faction is Faction.SPECTRA for card in player.discard_pile)

    def _has_domination(self, player: PlayerState, current: CardInstance) -> bool:
        factions = {
            card.definition.faction
            for card in [*player.hand, *player.play_zone]
            if card is not current
        }
        factions.update(
            card.definition.faction
            for card in player.champions
            if card is not current and card.instance_id in player.played_card_ids_this_turn
        )
        return {Faction.HOMODEUS, Faction.MAQUIS, Faction.SPECTRA} <= factions

    def _banishable_cards(self, player: PlayerState) -> list[CardInstance]:
        return [*player.hand, *player.discard_pile]

    def _banish_card(self, action: BanishCard) -> None:
        if self.state.phase not in (Phase.PLAY, Phase.BUY) or self.active.pending_banishes <= 0:
            raise InvalidActionError("BanishCard is not currently legal")
        player = self.active
        for zone in (player.hand, player.discard_pile):
            for index, card in enumerate(zone):
                if card.instance_id == action.card_id:
                    del zone[index]
                    player.pending_banishes -= 1
                    return
        raise InvalidActionError(
            f"Card is not in the active player's hand or discard pile: {action.card_id}"
        )

    def _skip_banish(self) -> None:
        if self.state.phase not in (Phase.PLAY, Phase.BUY) or self.active.pending_banishes <= 0:
            raise InvalidActionError("SkipBanish is not currently legal")
        self.active.pending_banishes = 0

    def _offer_free_recruit(
        self, player: PlayerState, maximum_cost: int, to_hand: bool
    ) -> None:
        if not any(
            card is not None and card.definition.cost <= maximum_cost
            for card in self.state.river
        ):
            return
        player.pending_free_recruit_cost = maximum_cost
        player.pending_free_recruit_to_hand = to_hand

    def _recruit_free_card(self, action: RecruitFreeCard) -> None:
        player = self.active
        if self.state.phase not in (Phase.PLAY, Phase.BUY) or player.pending_free_recruit_cost is None:
            raise InvalidActionError("RecruitFreeCard is not currently legal")
        if not 0 <= action.river_slot < len(self.state.river):
            raise InvalidActionError(f"Invalid river slot: {action.river_slot}")
        card = self.state.river[action.river_slot]
        if card is None or card.instance_id != action.card_instance_id:
            raise InvalidActionError("The selected river card is no longer available")
        if card.definition.cost > player.pending_free_recruit_cost:
            raise InvalidActionError("The selected card exceeds the free recruitment limit")

        player.pending_free_recruit_cost = None
        if self._arm_recruited_champion(player, card):
            pass
        elif player.pending_free_recruit_to_hand:
            player.hand.append(card)
        else:
            player.discard_pile.append(card)
        player.pending_free_recruit_to_hand = False
        self.state.river[action.river_slot] = (
            self.state.central_deck.pop() if self.state.central_deck else None
        )

    def _pass_play_phase(self) -> None:
        if self.state.phase is not Phase.PLAY:
            raise InvalidActionError(
                f"Action requires phase {Phase.PLAY.value}, got {self.state.phase.value}"
            )
        self.state.phase = Phase.BUY

    def _buy_card(self, action: BuyCard) -> None:
        if self.state.phase is not Phase.BUY:
            raise InvalidActionError(
                f"Action requires phase {Phase.BUY.value}, got {self.state.phase.value}"
            )
        if not 0 <= action.river_slot < len(self.state.river):
            raise InvalidActionError(f"Invalid river slot: {action.river_slot}")
        card = self.state.river[action.river_slot]
        if card is None:
            raise InvalidActionError(f"River slot is empty: {action.river_slot}")
        if card.instance_id != action.card_instance_id:
            raise InvalidActionError(
                f"River slot {action.river_slot} does not contain "
                f"{action.card_instance_id}"
            )
        if card.definition.cost > self.active.gems:
            raise InvalidActionError(
                f"Card cost {card.definition.cost} exceeds available Gems "
                f"{self.active.gems}"
            )

        self.active.gems -= card.definition.cost
        if not self._arm_recruited_champion(self.active, card):
            self.active.discard_pile.append(card)
        self.state.river[action.river_slot] = (
            self.state.central_deck.pop() if self.state.central_deck else None
        )

    def _recruit_mercenary(self, action: RecruitMercenary) -> None:
        if self.state.phase is not Phase.BUY:
            raise InvalidActionError(
                f"Action requires phase {Phase.BUY.value}, got {self.state.phase.value}"
            )
        if not 0 <= action.river_slot < len(self.state.river):
            raise InvalidActionError(f"Invalid river slot: {action.river_slot}")
        card = self.state.river[action.river_slot]
        if card is None or card.instance_id != action.card_instance_id:
            raise InvalidActionError("The selected river card is no longer available")
        if not card.definition.is_mercenary:
            raise InvalidActionError("The selected card is not a mercenary")
        if card.definition.cost > self.active.gems:
            raise InvalidActionError(
                f"Card cost {card.definition.cost} exceeds available Gems "
                f"{self.active.gems}"
            )

        self.active.gems -= card.definition.cost
        self.state.river[action.river_slot] = (
            self.state.central_deck.pop() if self.state.central_deck else None
        )
        self.active.play_zone.append(card)
        self.active.played_card_ids_this_turn.add(card.instance_id)
        self.active.recruited_mercenary_ids_this_turn.add(card.instance_id)
        self._resolve_card_effect(self.active, card)

    def _arm_recruited_champion(self, player: PlayerState, card: CardInstance) -> bool:
        if not (
            player.pending_homodeus_champion_recruitment
            and card.definition.is_champion
            and card.definition.faction is Faction.HOMODEUS
        ):
            return False
        player.pending_homodeus_champion_recruitment = False
        player.champions.append(card)
        if card.definition.on_play_effect is not None:
            self._resolve_effect(player, card, card.definition.on_play_effect)
        if card.definition.champion_ability is not None:
            self._resolve_champion_ability(player, card)
            player.activated_champion_ids.add(card.instance_id)
        return True

    def _stop_buying(self) -> None:
        if self.state.phase is not Phase.BUY:
            raise InvalidActionError(
                f"Action requires phase {Phase.BUY.value}, got {self.state.phase.value}"
            )
        self.active.gems = 0
        self.state.phase = Phase.ATTACK

    def _assign_power(self, action: AssignPower) -> None:
        if self.state.phase is not Phase.ATTACK:
            raise InvalidActionError(
                f"Action requires phase {Phase.ATTACK.value}, got {self.state.phase.value}"
            )
        active = self.active
        opponent = self.opponent
        if action.amount != active.power:
            raise InvalidActionError(
                f"Must assign exactly {active.power} Power"
            )
        if action.target != "opponent":
            if action.target not in self._legal_attack_champion_ids():
                raise InvalidActionError("The selected champion is not a legal attack target")
            target = next(
                card for card in opponent.champions if card.instance_id == action.target
            )
            target_health = target.definition.champion_health or 0
            self._destroy_champion(opponent, action.target)
            active.power -= target_health
            self._cleanup_if_attack_exhausted()
            return
        damage = max(0, action.amount - sum(card.definition.shield for card in opponent.hand))
        opponent.health -= damage
        if opponent.health <= 0:
            self.state.status = GameStatus.FINISHED
            self.state.winner = active.player_id
            return
        self._cleanup_and_start_next_turn()

    def _cleanup_if_attack_exhausted(self) -> None:
        if self.active.power <= 0:
            self._cleanup_and_start_next_turn()

    def _cleanup_and_start_next_turn(self) -> None:
        self.state.phase = Phase.CLEANUP
        active = self.active
        if active.recruited_mercenary_ids_this_turn:
            returned_mercenaries = []
            remaining_play_zone = []
            for card in active.play_zone:
                if card.instance_id in active.recruited_mercenary_ids_this_turn:
                    returned_mercenaries.append(card)
                else:
                    remaining_play_zone.append(card)
            active.play_zone = remaining_play_zone
            active.discard_pile.extend(active.play_zone)
            active.play_zone.clear()
            for card in returned_mercenaries:
                self.state.central_deck.insert(0, card)
        else:
            active.discard_pile.extend(active.play_zone)
            active.play_zone.clear()
        active.discard_pile.extend(active.hand)
        active.hand.clear()
        active.gems = 0
        active.power = 0
        active.activated_champion_ids.clear()
        active.played_card_ids_this_turn.clear()
        active.recruited_mercenary_ids_this_turn.clear()
        active.pending_decision = None
        active.pending_homodeus_champion_recruitment = False
        active.pending_banishes = 0
        active.pending_free_recruit_cost = None
        active.pending_free_recruit_to_hand = False
        active.mastery_action_used = False
        self.draw_many(active.player_id, self.STARTING_HAND_SIZE)
        self.state.active_player = active.player_id.opponent
        self.state.turn_number += 1
        self.state.phase = Phase.PLAY

    @property
    def active_player(self) -> PlayerId:
        return self.state.active_player
