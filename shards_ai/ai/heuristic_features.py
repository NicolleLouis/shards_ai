"""Feature extraction for the weighted heuristic policy.

The extractor deliberately uses only the detached observation and declarative card definitions.
It never applies actions or mutates game state.
"""

from __future__ import annotations

from dataclasses import fields, replace

from shards_ai.game.actions import (
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
    SkipBanish,
    StopBuying,
)
from shards_ai.game.cards import CardDefinition, CardInstance, Effect, Operation
from shards_ai.game.enums import Faction, Phase, PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.game import Game
from shards_ai.game.state import GameState, PlayerState

from .heuristic_evaluator import (
    ActionFeatures,
    CardAcquisitionWeights,
    CardConstraintWeights,
    HeuristicWeights,
)




ConditionFlags = tuple[bool, bool, bool, bool]
CommonConditionFlags = tuple[bool, bool, bool]


def is_win_card(card: CardInstance) -> bool:
    """Return whether a card has any declarative branch that wins the game."""

    effects = [card.definition.effect]
    if card.definition.on_play_effect is not None:
        effects.append(card.definition.on_play_effect)
    return any(
        operation.kind == "win"
        for effect in effects
        for step in effect.steps
        for operation in step.operations
    )


def is_replacement_card(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None = None,
) -> bool:
    """Return whether playing the card now produces an effective draw."""

    return _play_card_features(
        observation,
        player,
        card,
        constraint_weights,
        common_condition_flags,
    ).card_draw > 0.0


def _play_card_features(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None = None,
) -> ActionFeatures:
    """Evaluate a card's immediate effect independently of its current zone."""

    effect = (
        card.definition.on_play_effect
        if card.definition.is_champion and card.definition.on_play_effect is not None
        else card.definition.effect
    )
    features = _effect_features(
        observation, player, card, effect, constraint_weights, common_condition_flags
    )
    if card.definition.is_champion:
        features = replace(
            features,
            champion_value=float(card.definition.champion_health or 0) + 2.0,
            self_threat_delta=1.0,
        )
    return features


def features_for_action(
    observation: GameState,
    action: Action,
    player_id: PlayerId | None = None,
    acquisition_weights: CardAcquisitionWeights | None = None,
    constraint_weights: CardConstraintWeights | None = None,
    common_condition_flags: CommonConditionFlags | None = None,
    durable_replay_multiplier: float | None = None,
    heuristic_weights: HeuristicWeights | None = None,
) -> ActionFeatures:
    """Return the non-negative signals for one legal action."""

    player_id = player_id or observation.active_player
    acquisition_weights = acquisition_weights or CardAcquisitionWeights()
    constraint_weights = constraint_weights or CardConstraintWeights()
    player = observation.players[player_id]
    opponent = observation.players[player_id.opponent]
    common_condition_flags = common_condition_flags or _common_condition_flags(player)
    if isinstance(action, PlayCard):
        card = _find_card(player, action.card_id)
        if card is None:
            raise InvalidActionError(f"Card is not observable for action: {action.card_id}")
        return _play_card_features(
            observation, player, card, constraint_weights, common_condition_flags
        )

    if isinstance(action, BuyCard):
        card = _river_card(observation, action.river_slot, action.card_instance_id)
        return _purchase_features(
            observation,
            player,
            card,
            acquisition_weights,
            constraint_weights,
            common_condition_flags,
            immediate=False,
            apply_replay_factor=True,
            replay_multiplier=durable_replay_multiplier,
        )

    if isinstance(action, RecruitMercenary):
        card = _river_card(observation, action.river_slot, action.card_instance_id)
        return _purchase_features(
            observation,
            player,
            card,
            acquisition_weights,
            constraint_weights,
            common_condition_flags,
            immediate=True,
            apply_replay_factor=False,
            replay_multiplier=None,
        )

    if isinstance(action, RecruitFreeCard):
        card = _river_card(observation, action.river_slot, action.card_instance_id)
        features = _purchase_features(
            observation,
            player,
            card,
            acquisition_weights,
            constraint_weights,
            common_condition_flags,
            immediate=False,
            apply_replay_factor=True,
            replay_multiplier=durable_replay_multiplier,
        )
        return replace(features, cost_paid=0.0)

    if isinstance(action, ActivateChampion):
        champion = _find_card(player, action.champion_id, zones=("champions",))
        if champion is None or champion.definition.champion_ability is None:
            raise InvalidActionError(f"Champion is not observable for action: {action.champion_id}")
        return _ability_features(observation, player, champion, constraint_weights)

    if isinstance(action, GainMastery):
        heuristic_weights = heuristic_weights or HeuristicWeights()
        if (
            heuristic_weights.purchase_opportunity_cost == 0.0
            and heuristic_weights.mastery_threshold_value == 0.0
        ):
            return ActionFeatures(
                cost_paid=1.0,
                mastery_gained=1.0,
                mastery_advantage_delta=1.0 / 30.0,
            )
        purchase_delta, threshold_value = _gain_mastery_projection(
            observation,
            player_id,
            acquisition_weights,
            constraint_weights,
            heuristic_weights,
        )
        return ActionFeatures(
            cost_paid=1.0,
            mastery_gained=1.0,
            mastery_advantage_delta=1.0 / 30.0,
            purchase_opportunity_cost=purchase_delta,
            mastery_threshold_value=threshold_value,
        )

    if isinstance(action, BanishCard):
        card = _find_card(player, action.card_id, zones=("hand", "discard_pile"))
        if card is None:
            raise InvalidActionError(f"Card is not observable for action: {action.card_id}")
        return ActionFeatures(
            deck_thinning=acquisition_weights.banish_threshold
            - _card_acquisition_value(
                observation,
                player,
                card,
                acquisition_weights,
                constraint_weights,
                common_condition_flags,
            )
        )

    if isinstance(action, SkipBanish):
        return ActionFeatures(action_penalty=1.0)

    if isinstance(action, ChoosePendingDecision):
        return _pending_decision_features(
            observation,
            player,
            opponent,
            action.choice_id,
            acquisition_weights,
            constraint_weights,
            common_condition_flags,
        )

    if isinstance(action, AssignPower):
        return _attack_features(observation, player, opponent, action)

    if isinstance(action, (PassPlayPhase, StopBuying)):
        return ActionFeatures(phase_progress=1.0, action_penalty=1.0)

    raise InvalidActionError(f"Unsupported heuristic action: {action!r}")


def _gain_mastery_projection(
    observation: GameState,
    player_id: PlayerId,
    acquisition_weights: CardAcquisitionWeights,
    constraint_weights: CardConstraintWeights,
    heuristic_weights: HeuristicWeights,
) -> tuple[float, float]:
    """Project the cross-phase effects of spending one Gem for one Mastery."""

    player = observation.players[player_id]
    projected_mastery = min(30, player.mastery + 1)
    purchase_delta = 0.0
    if heuristic_weights.purchase_opportunity_cost != 0.0:
        before = _best_purchase_value(
            observation,
            player_id,
            player.gems,
            player.mastery,
            acquisition_weights,
            constraint_weights,
            heuristic_weights,
        )
        after = _best_purchase_value(
            observation,
            player_id,
            max(0, player.gems - 1),
            projected_mastery,
            acquisition_weights,
            constraint_weights,
            heuristic_weights,
        )
        purchase_delta = before - after

    threshold_value = 0.0
    if (
        heuristic_weights.mastery_threshold_value != 0.0
        and projected_mastery > player.mastery
    ):
        current_observation = _projected_observation(
            observation, player_id, mastery=player.mastery
        )
        next_observation = _projected_observation(
            observation, player_id, mastery=projected_mastery
        )
        for card in player.hand:
            action = PlayCard(card.instance_id)
            current_score = heuristic_weights.score(
                features_for_action(
                    current_observation,
                    action,
                    player_id,
                    acquisition_weights,
                    constraint_weights,
                    heuristic_weights=heuristic_weights,
                )
            )
            next_score = heuristic_weights.score(
                features_for_action(
                    next_observation,
                    action,
                    player_id,
                    acquisition_weights,
                    constraint_weights,
                    heuristic_weights=heuristic_weights,
                )
            )
            threshold_value += max(0.0, next_score - current_score)
    return purchase_delta, threshold_value


def _best_purchase_value(
    observation: GameState,
    player_id: PlayerId,
    gems: int,
    mastery: int,
    acquisition_weights: CardAcquisitionWeights,
    constraint_weights: CardConstraintWeights,
    heuristic_weights: HeuristicWeights,
) -> float:
    if gems <= 0:
        return 0.0
    player = observation.players[player_id]
    projected_player = replace(player, gems=gems, mastery=mastery)
    common_condition_flags = _common_condition_flags(projected_player)
    values: list[float] = []
    for card in observation.river:
        if card is None or card.definition.cost > gems:
            continue
        buy_features = _purchase_features(
            observation,
            projected_player,
            card,
            acquisition_weights,
            constraint_weights,
            common_condition_flags,
            immediate=False,
            apply_replay_factor=True,
            replay_multiplier=None,
        )
        values.append(heuristic_weights.score(buy_features))
        if card.definition.is_mercenary:
            recruit_features = _purchase_features(
                observation,
                projected_player,
                card,
                acquisition_weights,
                constraint_weights,
                common_condition_flags,
                immediate=True,
                apply_replay_factor=False,
                replay_multiplier=None,
            )
            values.append(heuristic_weights.score(recruit_features))
    return max(0.0, max(values, default=0.0))


def _projected_observation(
    observation: GameState,
    player_id: PlayerId,
    *,
    gems: int | None = None,
    mastery: int | None = None,
) -> GameState:
    player = observation.players[player_id]
    projected_player = replace(
        player,
        gems=player.gems if gems is None else gems,
        mastery=player.mastery if mastery is None else mastery,
    )
    players = dict(observation.players)
    players[player_id] = projected_player
    return replace(observation, players=players)


def _purchase_features(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    acquisition_weights: CardAcquisitionWeights,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags,
    *,
    immediate: bool,
    apply_replay_factor: bool,
    replay_multiplier: float | None,
) -> ActionFeatures:
    if not immediate:
        if not card.definition.is_champion and card.definition.on_play_effect is None:
            prospective = _prospective_effect_features(
                observation,
                player,
                card,
                card.definition.effect,
                constraint_weights,
                common_condition_flags,
                acquisition_weights,
            )
            acquisition_value = _weighted_acquisition_value(
                prospective, acquisition_weights
            )
            if apply_replay_factor:
                if replay_multiplier is None:
                    replay_multiplier = _durable_replay_multiplier(
                        observation, player, acquisition_weights.durable_replay_factor
                    )
                acquisition_value *= replay_multiplier
            return ActionFeatures(
                cost_paid=float(card.definition.cost),
                card_acquisition_value=acquisition_value,
                constraint_penalty=prospective.constraint_penalty,
            )
        acquisition_value = _card_acquisition_value(
            observation,
            player,
            card,
            acquisition_weights,
            constraint_weights,
            common_condition_flags,
        )
        if apply_replay_factor:
            if replay_multiplier is None:
                replay_multiplier = _durable_replay_multiplier(
                    observation, player, acquisition_weights.durable_replay_factor
                )
            acquisition_value *= replay_multiplier
        return ActionFeatures(
            cost_paid=float(card.definition.cost),
            card_acquisition_value=acquisition_value,
            constraint_penalty=_card_constraint_penalty(
                observation,
                player,
                card.definition,
                constraint_weights,
                common_condition_flags,
                acquisition_weights,
            ),
        )

    effect_features = _effect_features(
        observation,
        player,
        card,
        card.definition.effect,
        constraint_weights,
        common_condition_flags,
        effective_now=True,
    )
    return replace(
        effect_features,
        cost_paid=float(card.definition.cost),
        # A recruited mercenary is played immediately and does not become a
        # durable deck asset. Its value is represented by its immediate
        # effect, not by the card-acquisition estimate.
        card_acquisition_value=0.0,
        constraint_penalty=effect_features.constraint_penalty,
    )


def _durable_replay_multiplier(
    observation: GameState,
    player: PlayerState,
    factor: float,
) -> float:
    """Discount durable cards as the observable game progresses.

    Progress uses the strongest public end-game signal: mastery progress or
    health loss. The exponent is optimized, so zero means no discount and
    larger values make late purchases less attractive.
    """

    opponent = observation.players[player.player_id.opponent]
    mastery_progress = max(player.mastery, opponent.mastery) / 30.0
    health_progress = 1.0 - min(player.health, opponent.health) / 50.0
    progress = min(1.0, max(0.0, max(mastery_progress, health_progress)))
    opportunity = max(0.05, 1.0 - progress)
    return opportunity**factor


def _effect_features(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    effect: Effect,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None = None,
    *,
    effective_now: bool = False,
) -> ActionFeatures:
    operations = effect.operations_for_mastery(player.mastery)
    return _effect_features_for_operations(
        observation,
        player,
        card,
        effect,
        operations,
        constraint_weights,
        common_condition_flags,
        include_inactive=False,
        effective_now=effective_now,
    )


def _effect_features_for_operations(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    effect: Effect,
    operations: tuple[Operation, ...],
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None,
    *,
    include_inactive: bool,
    effective_now: bool = False,
) -> ActionFeatures:
    condition_flags = _condition_flags(player, card.definition, common_condition_flags)
    values: dict[str, float] = {
        "gems_produced": 0.0,
        "power_produced": 0.0,
        "health_advantage_delta": 0.0,
        "mastery_advantage_delta": 0.0,
        "opponent_threat_delta": 0.0,
    }
    penalty = 0.0
    terminal_win = 0.0
    projection_supported = True
    remaining_health = (
        max(0.0, float(Game.STARTING_HEALTH - player.health))
        if effective_now
        else None
    )

    for operation in operations:
        active = _operation_active(
            observation, player, card.definition, operation, condition_flags
        )
        if not active and not include_inactive:
            continue
        penalty += _operation_constraint_penalty(
            observation, player, card.definition, operation, condition_flags, constraint_weights
        )
        amount = float(operation.amount)
        if operation.kind == "gain_gems":
            values["gems_produced"] = values.get("gems_produced", 0) + amount
        elif operation.kind == "gain_power":
            values["power_produced"] = values.get("power_produced", 0) + amount
        elif operation.kind == "gain_mastery":
            values["mastery_gained"] = values.get("mastery_gained", 0) + amount
            values["mastery_advantage_delta"] += amount / 30.0
        elif operation.kind == "gain_health":
            effective_amount = (
                min(amount, remaining_health)
                if remaining_health is not None
                else amount
            )
            if remaining_health is not None:
                remaining_health -= effective_amount
            values["health_gained"] = values.get("health_gained", 0) + effective_amount
            values["health_advantage_delta"] += effective_amount / Game.STARTING_HEALTH
        elif operation.kind == "deal_damage":
            values["power_produced"] = values.get("power_produced", 0) + amount
            values["health_advantage_delta"] += amount / 50.0
        elif operation.kind == "draw_card":
            values["card_draw"] = values.get("card_draw", 0) + max(1.0, amount)
        elif operation.kind == "offer_banish":
            values["deck_thinning"] = values.get("deck_thinning", 0) + amount
        elif operation.kind in {"destroy_champion", "destroy_all_champions"}:
            values["opponent_threat_delta"] += (
                len(observation.players[player.player_id.opponent].champions)
                if operation.kind == "destroy_all_champions"
                else 1.0
            )
        elif operation.kind in {"recover_champion", "recover_mercenary"}:
            values["card_draw"] = values.get("card_draw", 0) + 1.0
        elif operation.kind == "win":
            terminal_win = 1.0
        else:
            projection_supported = False

    return ActionFeatures(
        gems_produced=values.get("gems_produced", 0),
        power_produced=values.get("power_produced", 0),
        mastery_gained=values.get("mastery_gained", 0),
        health_gained=values.get("health_gained", 0),
        card_draw=values.get("card_draw", 0),
        deck_thinning=values.get("deck_thinning", 0),
        constraint_penalty=penalty,
        terminal_win=terminal_win,
        health_advantage_delta=values.get("health_advantage_delta", 0),
        mastery_advantage_delta=values.get("mastery_advantage_delta", 0),
        opponent_threat_delta=values.get("opponent_threat_delta", 0),
        projection_supported=projection_supported,
    )


def _prospective_effect_features(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    effect: Effect,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None = None,
    acquisition_weights: CardAcquisitionWeights | None = None,
) -> ActionFeatures:
    """Estimate a durable card by aggregating current and future effect branches."""

    if not effect.steps:
        return _effect_features(
            observation, player, card, effect, constraint_weights, common_condition_flags
        )

    active_index = next(
        (
            index
            for index, step in enumerate(effect.steps)
            if step.mastery_at_least is None or player.mastery >= step.mastery_at_least
        ),
        None,
    )
    acquisition_weights = acquisition_weights or CardAcquisitionWeights()
    branches = []
    for index, step in enumerate(effect.steps):
        operation_features = []
        for operation in step.operations:
            operation_feature = _effect_features_for_operations(
                observation,
                player,
                card,
                effect,
                (operation,),
                constraint_weights,
                common_condition_flags,
                include_inactive=True,
            )
            operation_value = _weighted_acquisition_value(
                operation_feature, acquisition_weights
            )
            if operation_value <= operation_feature.constraint_penalty:
                operation_feature = replace(
                    ActionFeatures(),
                    projection_supported=operation_feature.projection_supported,
                )
            operation_features.append(operation_feature)
        branch = _sum_action_features(operation_features)
        if index != active_index and step.mastery_at_least is not None:
            branch = replace(
                branch,
                constraint_penalty=branch.constraint_penalty
                + constraint_weights.mastery * step.mastery_at_least / 20.0
                + max(0, step.mastery_at_least - player.mastery)
                * constraint_weights.mastery
                / 10.0,
            )
        if _weighted_acquisition_value(branch, acquisition_weights) <= branch.constraint_penalty:
            branch = replace(
                ActionFeatures(),
                projection_supported=branch.projection_supported,
            )
        branches.append(branch)

    return _sum_action_features(branches)


def _sum_action_features(features: list[ActionFeatures]) -> ActionFeatures:
    if not features:
        return ActionFeatures()
    totals = {
        field.name: sum(float(getattr(feature, field.name)) for feature in features)
        for field in fields(ActionFeatures)
        if field.name != "projection_supported"
    }
    totals["projection_supported"] = all(feature.projection_supported for feature in features)
    return ActionFeatures(**totals)


def _ability_features(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    constraint_weights: CardConstraintWeights,
) -> ActionFeatures:
    ability = card.definition.champion_ability
    assert ability is not None
    power_produced = 0.0
    mastery_gained = 0.0
    card_draw = 0.0
    gems_produced = 0.0
    health_gained = 0.0
    if ability.kind in {"gain_power", "gain_power_per_played_faction", "gain_power_threshold", "gain_power_then_recover_faction"}:
        power_produced = float(ability.amount)
    if ability.kind in {"gain_mastery_then_draw", "gain_mastery_if_domination"}:
        mastery_gained = float(ability.amount)
    if ability.kind in {"gain_mastery_then_draw", "draw_if_domination", "draw_if_champion_faction_count"}:
        card_draw = float(max(1, ability.draw_amount))
    if ability.kind == "gain_gem_and_arm_recruitment":
        gems_produced = float(ability.amount)
    if ability.kind == "gain_health_per_champion":
        count = sum(candidate.definition.faction is ability.faction for candidate in player.champions)
        health_gained = float(ability.amount * count)
    if ability.kind == "gain_power_per_champion":
        count = sum(candidate.definition.faction is ability.faction for candidate in player.champions)
        power_produced = float(ability.amount * count)
    if ability.kind == "gain_gems_then_copy_faction":
        gems_produced = float(ability.amount)
    threshold_penalty = 0.0
    if ability.threshold is not None:
        threshold_penalty += constraint_weights.mastery * ability.threshold / 20.0
        if player.mastery < ability.threshold:
            threshold_penalty += constraint_weights.mastery * (ability.threshold - player.mastery) / 10.0
    return ActionFeatures(
        gems_produced=gems_produced,
        power_produced=power_produced,
        mastery_gained=mastery_gained,
        card_draw=card_draw,
        health_gained=health_gained,
        constraint_penalty=threshold_penalty,
        mastery_advantage_delta=mastery_gained / 30.0,
        health_advantage_delta=health_gained / 50.0,
    )


def _pending_decision_features(
    observation: GameState,
    player: PlayerState,
    opponent: PlayerState,
    choice_id: str,
    acquisition_weights: CardAcquisitionWeights,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags,
) -> ActionFeatures:
    decision = player.pending_decision
    if decision is None:
        raise InvalidActionError("No pending decision is available")
    if decision.kind in {"destroy_opponent_champion", "destroy_all_champions"}:
        target = _find_card(opponent, choice_id, zones=("champions",))
        if target is None:
            raise InvalidActionError(f"Champion is not observable for decision: {choice_id}")
        return replace(_champion_target_features(target), opponent_threat_delta=1.0)
    target = _find_card(player, choice_id)
    if target is not None:
        return ActionFeatures(
            card_acquisition_value=_card_acquisition_value(
                observation,
                player,
                target,
                acquisition_weights,
                constraint_weights,
                common_condition_flags,
            )
        )
    return ActionFeatures(action_penalty=1.0)


def _attack_features(
    observation: GameState,
    player: PlayerState,
    opponent: PlayerState,
    action: AssignPower,
) -> ActionFeatures:
    if action.target == "opponent":
        shield = sum(card.definition.shield for card in opponent.hand)
        damage = max(0, action.amount - shield)
        return ActionFeatures(
            damage_value=float(damage),
            lethal=float(damage >= opponent.health),
            health_advantage_delta=float(damage) / 50.0,
        )
    target = _find_card(opponent, action.target, zones=("champions",))
    if target is None:
        raise InvalidActionError(f"Champion is not observable for action: {action.target}")
    return replace(_champion_target_features(target), opponent_threat_delta=1.0)


def _champion_target_features(card: CardInstance) -> ActionFeatures:
    definition = card.definition
    ability_value = 2.0 if definition.champion_ability is not None else 0.0
    denial = 1.0 if definition.passive_kind is not None or definition.champion_ability is not None else 0.0
    return ActionFeatures(
        champion_value=float(definition.champion_health or 0) + ability_value,
        target_denial=denial,
        opponent_threat_delta=1.0,
    )


def _card_acquisition_value(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    acquisition_weights: CardAcquisitionWeights,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None = None,
) -> float:
    features = _prospective_effect_features(
        observation,
        player,
        card,
        card.definition.effect,
        constraint_weights,
        common_condition_flags,
        acquisition_weights,
    )
    value = _weighted_acquisition_value(features, acquisition_weights)
    if card.definition.is_champion:
        value += float(card.definition.champion_health or 0) + 2.0
        if card.definition.on_play_effect is not None:
            on_play = _prospective_effect_features(
                observation,
                player,
                card,
                card.definition.on_play_effect,
                constraint_weights,
                common_condition_flags,
                acquisition_weights,
            )
            value += (
                on_play.gems_produced * acquisition_weights.gems_produced
                + on_play.power_produced * acquisition_weights.power_produced
                + on_play.mastery_gained * acquisition_weights.mastery_gained
                + on_play.health_gained * acquisition_weights.health_gained
                + on_play.card_draw * acquisition_weights.card_draw
            )
        if card.definition.champion_ability is not None:
            ability = _ability_features(
                observation,
                player,
                card,
                constraint_weights,
            )
            value += _weighted_acquisition_value(ability, acquisition_weights)
    value += float(card.definition.shield) * 0.5
    return max(0.0, value)


def _weighted_acquisition_value(
    features: ActionFeatures,
    acquisition_weights: CardAcquisitionWeights,
) -> float:
    return (
        features.gems_produced * acquisition_weights.gems_produced
        + features.power_produced * acquisition_weights.power_produced
        + features.mastery_gained * acquisition_weights.mastery_gained
        + features.health_gained * acquisition_weights.health_gained
        + features.card_draw * acquisition_weights.card_draw
        + features.deck_thinning * acquisition_weights.deck_thinning
        + features.target_denial * acquisition_weights.target_denial
    )


def _card_constraint_penalty(
    observation: GameState,
    player: PlayerState,
    definition: CardDefinition,
    constraint_weights: CardConstraintWeights,
    common_condition_flags: CommonConditionFlags | None = None,
    acquisition_weights: CardAcquisitionWeights | None = None,
) -> float:
    card = CardInstance(instance_id="__prospective__", definition=definition)
    penalty = _prospective_effect_features(
        observation,
        player,
        card,
        definition.effect,
        constraint_weights,
        common_condition_flags,
        acquisition_weights,
    ).constraint_penalty
    if definition.on_play_effect is not None:
        penalty += _prospective_effect_features(
            observation,
            player,
            card,
            definition.on_play_effect,
            constraint_weights,
            common_condition_flags,
            acquisition_weights,
        ).constraint_penalty
    return penalty


def _operation_constraint_penalty(
    observation: GameState,
    player: PlayerState,
    definition: CardDefinition,
    operation: Operation,
    condition_flags: ConditionFlags | None = None,
    constraint_weights: CardConstraintWeights | None = None,
) -> float:
    condition_flags = condition_flags or _condition_flags(player, definition)
    constraint_weights = constraint_weights or CardConstraintWeights()
    penalty = 0.0
    if operation.mastery_at_least is not None:
        penalty += constraint_weights.mastery * operation.mastery_at_least / 20.0
        if player.mastery < operation.mastery_at_least:
            penalty += constraint_weights.mastery * (operation.mastery_at_least - player.mastery) / 10.0
    if operation.requires_union and not condition_flags[0]:
        penalty += constraint_weights.union
    if operation.requires_echo and not condition_flags[1]:
        penalty += constraint_weights.echo
    if operation.requires_domination and not condition_flags[2]:
        penalty += constraint_weights.domination
    if operation.requires_inspiration and not condition_flags[3]:
        penalty += constraint_weights.inspiration
    if operation.health_at_least is not None and player.health < operation.health_at_least:
        penalty += constraint_weights.health * (operation.health_at_least - player.health) / 10.0
    return penalty


def _operation_active(
    observation: GameState,
    player: PlayerState,
    definition: CardDefinition,
    operation: Operation,
    condition_flags: ConditionFlags | None = None,
) -> bool:
    condition_flags = condition_flags or _condition_flags(player, definition)
    if operation.mastery_at_least is not None and player.mastery < operation.mastery_at_least:
        return False
    if operation.health_at_least is not None and player.health < operation.health_at_least:
        return False
    return (
        (not operation.requires_union or condition_flags[0])
        and (not operation.requires_echo or condition_flags[1])
        and (not operation.requires_domination or condition_flags[2])
        and (not operation.requires_inspiration or condition_flags[3])
    )


def _condition_flags(
    player: PlayerState,
    definition: CardDefinition,
    common: CommonConditionFlags | None = None,
) -> ConditionFlags:
    common = common or _common_condition_flags(player)
    return (_has_union(player, definition), *common)


def _common_condition_flags(player: PlayerState) -> CommonConditionFlags:
    return (
        _has_echo(player),
        _has_domination(player),
        bool(player.champions),
    )


def _has_union(player: PlayerState, definition: CardDefinition) -> bool:
    for zone in (player.hand, player.play_zone, player.champions):
        for card in zone:
            if card.definition is not definition and card.definition.faction is definition.faction:
                return True
    return False


def _has_echo(player: PlayerState) -> bool:
    for card in player.discard_pile:
        if card.definition.faction is Faction.SPECTRA:
            return True
    return False


def _has_domination(player: PlayerState) -> bool:
    has_homodeus = has_maquis = has_spectra = False
    for zone in (player.hand, player.play_zone, player.champions):
        for card in zone:
            faction = card.definition.faction
            has_homodeus = has_homodeus or faction is Faction.HOMODEUS
            has_maquis = has_maquis or faction is Faction.MAQUIS
            has_spectra = has_spectra or faction is Faction.SPECTRA
            if has_homodeus and has_maquis and has_spectra:
                return True
    return False


def _find_card(
    player: PlayerState,
    instance_id: str,
    zones: tuple[str, ...] = ("hand", "play_zone", "champions", "discard_pile"),
) -> CardInstance | None:
    for zone_name in zones:
        for card in getattr(player, zone_name):
            if card.instance_id == instance_id:
                return card
    return None


def _river_card(observation: GameState, slot: int, instance_id: str) -> CardInstance:
    if not 0 <= slot < len(observation.river):
        raise InvalidActionError(f"Invalid river slot: {slot}")
    card = observation.river[slot]
    if card is None or card.instance_id != instance_id:
        raise InvalidActionError(f"River card is not observable: {instance_id}")
    return card
