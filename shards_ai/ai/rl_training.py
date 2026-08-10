"""Online PPO training utilities for the action-conditioned neural player."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId
from shards_ai.game.actions import Action
from shards_ai.game.observation import NeuralObservation

from .action_representation import ActionRepresentation, representation_for_neural_action
from .card_value_shaping import deckbuilding_shaping_delta, load_card_values
from .heuristic_player import HeuristicPlayer
from .heuristic_profiles import load_profile
from .neural_model import NeuralActionScorer, NeuralModelConfig, build_neural_scorer
from .neural_player import NeuralPlayer
from .neural_training_profiles import NeuralTrainingProfile
from .random_player import RandomPlayer


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    """One learner decision retained in memory until a PPO update completes."""

    episode_id: int
    game_seed: int
    opponent_id: str
    neural_player_id: PlayerId
    turn_number: int
    observation: NeuralObservation
    legal_action_representations: tuple[ActionRepresentation, ...]
    chosen_action_index: int
    old_log_probability: float
    value_estimate: float
    reward: float = 0.0
    done: bool = False


@dataclass(frozen=True, slots=True)
class RolloutResult:
    transitions: tuple[RolloutTransition, ...]
    games: int
    games_by_opponent: Mapping[str, int]
    outcomes_by_opponent: Mapping[str, Mapping[str, int]]
    transitions_by_episode: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    reference_kl: float
    advantages_mean: float
    transitions: int


class NeuralActorCritic(nn.Module):
    """Actor-critic for either independent or globally contextualized action logits."""

    def __init__(
        self,
        config: NeuralModelConfig | None = None,
        *,
        architecture: str = "independent_action",
    ) -> None:
        super().__init__()
        self.backbone = build_neural_scorer(architecture, config)
        self.architecture = architecture
        self.config = self.backbone.config
        self.policy_head = copy.deepcopy(self.backbone.scorer)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.config.state_hidden_dim),
            nn.Linear(self.config.state_hidden_dim, self.config.state_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.state_hidden_dim, 1),
        )
        # The v001 encoder can have large activations. Start the critic at the
        # neutral terminal-reward estimate instead of injecting a huge random
        # bootstrap value into the first GAE rollout.
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def card_ids(self) -> tuple[str, ...]:
        return self.backbone.card_ids

    def policy_logits(
        self,
        observation: NeuralObservation,
        actions: Sequence[ActionRepresentation],
    ) -> Tensor:
        if not actions:
            return torch.empty(0, device=self.device)
        state, action, context = self._encode_inputs(observation, actions)
        state_batch = state.expand(len(actions), -1)
        return self.policy_head(self._policy_features(state_batch, action, context)).squeeze(1)

    def _encode_inputs(
        self,
        observation: NeuralObservation,
        actions: Sequence[ActionRepresentation],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if not actions:
            raise ValueError("Cannot encode an empty legal action list")
        card_ids = list(self.backbone._observation_card_ids(observation))
        card_ids.extend(action.card_definition_id for action in actions)
        embedding_lookup = self.backbone._embedding_lookup(card_ids)
        state = self.backbone.encode_observation(observation, embedding_lookup=embedding_lookup)
        action = self.backbone.encode_actions(
            actions, observation=observation, embedding_lookup=embedding_lookup
        )
        context = None
        if self.architecture == "global_candidate_context":
            context = self.backbone.candidate_context_encoder(action.mean(dim=0, keepdim=True))
        return state, action, context

    @staticmethod
    def _policy_features(state: Tensor, action: Tensor, context: Tensor | None) -> Tensor:
        if context is not None:
            context = context.expand(action.shape[0], -1)
            return torch.cat((state, action, context), dim=1)
        return torch.cat((state, action), dim=1)

    def evaluate(
        self,
        observation: NeuralObservation,
        actions: Sequence[ActionRepresentation],
    ) -> tuple[Tensor, Tensor]:
        """Return policy logits and value while encoding the observation once."""
        state, action, context = self._encode_inputs(observation, actions)
        state_batch = state.expand(len(actions), -1)
        logits = self.policy_head(self._policy_features(state_batch, action, context)).squeeze(1)
        return logits, self.value_from_state(state)

    def value(self, observation: NeuralObservation) -> Tensor:
        card_ids = self.backbone._observation_card_ids(observation)
        state = self.backbone.encode_observation(observation, embedding_lookup=self.backbone._embedding_lookup(card_ids))
        return self.value_from_state(state)

    def value_from_state(self, state: Tensor) -> Tensor:
        return self.value_head(state).squeeze(-1)

    def forward(
        self,
        observation: NeuralObservation,
        actions: Sequence[ActionRepresentation],
    ) -> Tensor:
        """Return policy logits so the model remains usable by ``NeuralPlayer``."""
        return self.policy_logits(observation, actions)

    def inference_state_dict(self) -> dict[str, Tensor]:
        """Export the actor as a v001-compatible ``NeuralActionScorer`` state dict."""
        state = dict(self.backbone.state_dict())
        for key, value in self.policy_head.state_dict().items():
            state[f"scorer.{key}"] = value
        return state

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, object],
        *,
        config: NeuralModelConfig | None = None,
    ) -> "NeuralActorCritic":
        model_config = config or NeuralModelConfig(**checkpoint["model_config"])
        architecture = str(checkpoint.get("architecture", "independent_action"))
        model = cls(model_config, architecture=architecture)
        actor_state = checkpoint.get("actor_critic_state_dict")
        if actor_state is not None:
            model.load_state_dict(actor_state)
            return model
        scorer_state = checkpoint.get("model_state_dict")
        if scorer_state is None:
            raise ValueError("Checkpoint does not contain a neural model state")
        model.backbone.load_state_dict(scorer_state)
        model.policy_head.load_state_dict(model.backbone.scorer.state_dict())
        return model


@dataclass(frozen=True, slots=True)
class _DecisionPayload:
    observation: NeuralObservation
    legal_action_representations: tuple[ActionRepresentation, ...]
    chosen_action_index: int
    old_log_probability: float
    value_estimate: float
    turn_number: int


class PPOTrainingPlayer:
    """Stochastic learner policy that exposes metadata for the rollout collector."""

    observation_kind = "neural"
    observation_is_read_only = True

    def __init__(
        self,
        player_id: PlayerId,
        model: NeuralActorCritic,
        *,
        torch_generator: torch.Generator | None = None,
    ) -> None:
        self.player_id = player_id
        self.model = model
        self.torch_generator = torch_generator
        self._last_decision: _DecisionPayload | None = None

    def choose_action(
        self,
        observation: NeuralObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("Cannot choose an action from an empty action list")
        representations = tuple(
            representation_for_neural_action(action, observation) for action in actions
        )
        with torch.no_grad():
            logits, value = self.model.evaluate(observation, representations)
            distribution = Categorical(logits=logits)
            if self.torch_generator is None:
                chosen_tensor = distribution.sample()
            else:
                chosen_tensor = torch.multinomial(
                    distribution.probs,
                    1,
                    generator=self.torch_generator,
                ).squeeze(0)
            chosen = int(chosen_tensor.item())
            log_probability = float(distribution.log_prob(chosen_tensor).item())
            value_estimate = float(value.item())
        self._last_decision = _DecisionPayload(
            observation=observation,
            legal_action_representations=representations,
            chosen_action_index=chosen,
            old_log_probability=log_probability,
            value_estimate=value_estimate,
            turn_number=observation.turn_number,
        )
        return actions[chosen]

    def pop_last_decision(self) -> _DecisionPayload:
        if self._last_decision is None:
            raise RuntimeError("No learner decision is available for the current transition")
        payload = self._last_decision
        self._last_decision = None
        return payload


def terminal_reward(state, player_id: PlayerId) -> float:
    """Return only the terminal outcome from the learner's perspective."""
    if state.status is GameStatus.DRAW or state.winner is None:
        return 0.0
    return 1.0 if state.winner is player_id else -1.0


def choose_opponent(
    rng: GameRandom,
    opponents: Mapping[str, float],
) -> str:
    available = [(name, float(weight)) for name, weight in opponents.items() if weight > 0]
    if not available:
        raise ValueError("At least one opponent must have a positive weight")
    total = sum(weight for _name, weight in available)
    threshold = rng.random() * total
    cumulative = 0.0
    for name, weight in available:
        cumulative += weight
        if threshold < cumulative:
            return name
    return available[-1][0]


def _collect_episode(
    model: NeuralActorCritic,
    profile: NeuralTrainingProfile,
    game_index: int,
    heuristic_profiles: Mapping[str, object],
    card_values: Mapping[str, float] | None = None,
    shaping_beta: float = 0.0,
    shaping_clip: float = 1.0,
) -> tuple[str, tuple[RolloutTransition, ...], float]:
    game_seed = profile.seed * 1_000_003 + game_index
    root_rng = GameRandom(game_seed)
    opponent_name = choose_opponent(root_rng.derive("opponent"), profile.opponents)
    neural_id = PlayerId.PLAYER_1 if game_index % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = neural_id.opponent
    game = Game.new(seed=game_seed, rng=root_rng.derive("engine"))
    policy_generator = torch.Generator(device="cpu")
    policy_generator.manual_seed(game_seed % (2**63 - 1))
    learner = PPOTrainingPlayer(neural_id, model, torch_generator=policy_generator)
    if opponent_name == "random":
        opponent = RandomPlayer(opponent_id, root_rng.derive("random-opponent"))
    elif opponent_name in heuristic_profiles:
        opponent_profile = heuristic_profiles[opponent_name]
        opponent = HeuristicPlayer(
            opponent_id,
            opponent_profile.weights,
            opponent_profile.card_acquisition_weights,
            opponent_profile.constraint_weights,
        )
    else:
        raise ValueError(f"Unsupported RL opponent: {opponent_name!r}")
    runner = GameRunner(
        game,
        {neural_id: learner, opponent_id: opponent},
        max_actions=profile.max_actions,
        max_turns=profile.max_turns,
    )
    episode: list[RolloutTransition] = []

    def on_decision(_observation, _legal_actions, _chosen, player_id) -> None:
        if player_id is not neural_id:
            return
        payload = learner.pop_last_decision()
        episode.append(RolloutTransition(
            episode_id=game_index,
            game_seed=game_seed,
            opponent_id=opponent_name,
            neural_player_id=neural_id,
            turn_number=payload.turn_number,
            observation=payload.observation,
            legal_action_representations=payload.legal_action_representations,
            chosen_action_index=payload.chosen_action_index,
            old_log_probability=payload.old_log_probability,
            value_estimate=payload.value_estimate,
        ))

    def on_transition(before, action, after, player_id) -> None:
        if card_values is None or player_id is not neural_id:
            return
        if not episode:
            raise RuntimeError("Reward shaping transition has no matching decision")
        delta = shaping_beta * deckbuilding_shaping_delta(
            before, after, action, player_id, card_values, clip=shaping_clip
        )
        if delta:
            episode[-1] = replace(episode[-1], reward=episode[-1].reward + delta)

    final_state = runner.run(
        decision_observer=on_decision,
        transition_observer=on_transition,
        observer_before_state_factory=game.shaping_observation_for if card_values is not None else None,
    )
    if not episode:
        raise RuntimeError(f"RL episode {game_index} produced no learner transition")
    reward = terminal_reward(final_state, neural_id)
    episode[-1] = replace(episode[-1], reward=episode[-1].reward + reward, done=True)
    return opponent_name, tuple(episode), reward


def collect_rollout(
    model: NeuralActorCritic,
    profile: NeuralTrainingProfile,
    *,
    start_game_index: int,
    games: int | None = None,
    max_transitions: int | None = None,
    workers: int = 1,
) -> RolloutResult:
    """Play complete episodes and collect only learner transitions in memory."""
    requested_games = games if games is not None else profile.games_per_update
    if requested_games <= 0:
        raise ValueError("games must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    heuristic_profiles = {
        name: load_profile(Path(f"configs/heuristic_profiles/{name}.yaml"))
        for name in ("v007", "v008")
        if name in profile.opponents
    }
    shaping_config = dict(profile.reward_shaping or {})
    card_values = None
    shaping_beta = 0.0
    shaping_clip = 1.0
    if shaping_config.get("enabled", False):
        shaping_beta = float(shaping_config.get("beta", 0.05))
        shaping_clip = float(shaping_config.get("clip", 1.0))
        if shaping_beta < 0 or shaping_clip <= 0:
            raise ValueError("Reward shaping beta must be non-negative and clip must be positive")
        card_values = load_card_values(profile.resolve_path(str(shaping_config.get(
            "card_values_path",
            "configs/neural_training_profiles/card_values_v008.yaml",
        ))))
    transitions: list[RolloutTransition] = []
    games_by_opponent: dict[str, int] = {}
    outcomes_by_opponent: dict[str, dict[str, int]] = {}
    transitions_by_episode: list[int] = []

    def consume(result: tuple[str, tuple[RolloutTransition, ...], float]) -> None:
        opponent_name, episode, reward = result
        transitions.extend(episode)
        transitions_by_episode.append(len(episode))
        games_by_opponent[opponent_name] = games_by_opponent.get(opponent_name, 0) + 1
        outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        outcomes = outcomes_by_opponent.setdefault(opponent_name, {"win": 0, "loss": 0, "draw": 0})
        outcomes[outcome] += 1

    game_indices = range(start_game_index, start_game_index + requested_games)
    if workers == 1:
        results = (
            _collect_episode(
                model, profile, game_index, heuristic_profiles,
                card_values, shaping_beta, shaping_clip,
            )
            for game_index in game_indices
        )
        for result in results:
            if max_transitions is not None and transitions and len(transitions) >= max_transitions:
                break
            consume(result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch_start in range(0, requested_games, workers):
                batch_indices = list(
                    range(start_game_index + batch_start,
                          min(start_game_index + batch_start + workers, start_game_index + requested_games))
                )
                for result in executor.map(
                    lambda game_index: _collect_episode(
                        model, profile, game_index, heuristic_profiles,
                        card_values, shaping_beta, shaping_clip,
                    ),
                    batch_indices,
                ):
                    if max_transitions is not None and transitions and len(transitions) >= max_transitions:
                        break
                    consume(result)
                if max_transitions is not None and transitions and len(transitions) >= max_transitions:
                    break
    return RolloutResult(
        transitions=tuple(transitions),
        games=len(transitions_by_episode),
        games_by_opponent=games_by_opponent,
        outcomes_by_opponent=outcomes_by_opponent,
        transitions_by_episode=tuple(transitions_by_episode),
    )


def evaluate_greedy_model(
    model: NeuralActorCritic,
    profile: NeuralTrainingProfile,
) -> dict[str, object]:
    """Evaluate the current actor greedily on a fixed panel of opponents."""
    scorer = build_neural_scorer(model.architecture, model.config)
    scorer.load_state_dict(model.inference_state_dict())
    scorer.eval()
    heuristic_profiles = {
        name: load_profile(Path(f"configs/heuristic_profiles/{name}.yaml"))
        for name in ("v007", "v008")
    }
    by_opponent: dict[str, dict[str, float | int]] = {}
    for opponent_name in ("random", "v007", "v008"):
        wins = losses = draws = 0
        for index in range(profile.evaluation_games):
            seed = profile.evaluation_seed + index
            root_rng = GameRandom(seed)
            game = Game.new(seed=seed, rng=root_rng.derive("engine"))
            neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
            opponent_id = neural_id.opponent
            neural = NeuralPlayer(neural_id, None, root_rng.derive("neural"), scorer=scorer)
            if opponent_name == "random":
                opponent = RandomPlayer(opponent_id, root_rng.derive("opponent"))
            else:
                opponent_profile = heuristic_profiles[opponent_name]
                opponent = HeuristicPlayer(
                    opponent_id,
                    opponent_profile.weights,
                    opponent_profile.card_acquisition_weights,
                    opponent_profile.constraint_weights,
                )
            state = GameRunner(
                game,
                {neural_id: neural, opponent_id: opponent},
                max_actions=profile.max_actions,
                max_turns=profile.max_turns,
            ).run()
            if state.winner is neural_id:
                wins += 1
            elif state.winner is opponent_id:
                losses += 1
            else:
                draws += 1
        games = wins + losses + draws
        by_opponent[opponent_name] = {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / games if games else 0.0,
        }
    score = weighted_evaluation_score({"by_opponent": by_opponent}, profile.opponents)
    return {"score": score, "by_opponent": by_opponent}


def weighted_evaluation_score(
    evaluation: Mapping[str, object],
    opponents: Mapping[str, float],
) -> float:
    """Score an evaluation with the profile's opponent mix."""
    by_opponent = evaluation.get("by_opponent", {})
    if not isinstance(by_opponent, Mapping):
        raise ValueError("Evaluation must contain a by_opponent mapping")
    weighted_rates = []
    for name, raw_weight in opponents.items():
        weight = float(raw_weight)
        result = by_opponent.get(name)
        if weight > 0:
            if not isinstance(result, Mapping):
                raise ValueError(f"Evaluation is missing opponent {name!r}")
            weighted_rates.append((weight, float(result["win_rate"])))
    if not weighted_rates:
        raise ValueError("Evaluation has no opponent with a positive profile weight")
    total_weight = sum(weight for weight, _rate in weighted_rates)
    return sum(weight * rate for weight, rate in weighted_rates) / total_weight


def is_monotonic_evaluation_improvement(
    candidate: Mapping[str, object],
    incumbent: Mapping[str, object],
    opponents: Mapping[str, float],
    *,
    tolerated_opponents: Sequence[str] = (),
    tolerance_rate: float = 0.0,
) -> bool:
    """Return whether a candidate improves within optional per-opponent tolerances."""
    candidate_by_opponent = candidate.get("by_opponent", {})
    incumbent_by_opponent = incumbent.get("by_opponent", {})
    if not isinstance(candidate_by_opponent, Mapping) or not isinstance(incumbent_by_opponent, Mapping):
        raise ValueError("Evaluations must contain by_opponent mappings")
    for name, weight in opponents.items():
        if float(weight) <= 0:
            continue
        candidate_result = candidate_by_opponent.get(name)
        incumbent_result = incumbent_by_opponent.get(name)
        if not isinstance(candidate_result, Mapping) or not isinstance(incumbent_result, Mapping):
            raise ValueError(f"Evaluation is missing opponent {name!r}")
        tolerance = tolerance_rate if name in tolerated_opponents else 0.0
        if float(candidate_result["win_rate"]) + 1e-12 < float(incumbent_result["win_rate"]) - tolerance:
            return False
    return weighted_evaluation_score(candidate, opponents) > weighted_evaluation_score(incumbent, opponents)


def gae_returns(
    transitions: Sequence[RolloutTransition],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Calculate GAE without carrying credit across episode boundaries."""
    advantages = torch.zeros(len(transitions), dtype=torch.float32)
    returns = torch.zeros(len(transitions), dtype=torch.float32)
    running = 0.0
    next_value = 0.0
    next_episode_id: int | None = None
    for index in reversed(range(len(transitions))):
        transition = transitions[index]
        if next_episode_id is not None and transition.episode_id != next_episode_id:
            running = 0.0
            next_value = 0.0
        non_terminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * non_terminal - transition.value_estimate
        running = delta + gamma * gae_lambda * non_terminal * running
        advantages[index] = running
        returns[index] = running + transition.value_estimate
        next_value = transition.value_estimate
        next_episode_id = transition.episode_id
    return advantages, returns


def ppo_update(
    model: NeuralActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[RolloutTransition],
    *,
    optimization_epochs: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_loss_coefficient: float,
    entropy_coefficient: float,
    reference_model: NeuralActorCritic | None = None,
    reference_kl_coefficient: float = 0.0,
) -> PPOUpdateMetrics:
    if not transitions:
        raise ValueError("Cannot optimize an empty rollout")
    advantages, returns = gae_returns(transitions, gamma=gamma, gae_lambda=gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    reference_log_probabilities: list[Tensor] | None = None
    if reference_model is not None and reference_kl_coefficient > 0:
        reference_model.eval()
        reference_log_probabilities = []
        with torch.no_grad():
            for transition in transitions:
                reference_logits, _value = reference_model.evaluate(
                    transition.observation,
                    transition.legal_action_representations,
                )
                reference_log_probabilities.append(torch.log_softmax(reference_logits, dim=0))
    metrics: list[tuple[float, float, float, float, float, float]] = []
    model.train()
    for _epoch in range(optimization_epochs):
        for batch_indices in torch.randperm(len(transitions)).split(minibatch_size):
            policy_losses = []
            value_losses = []
            entropies = []
            ratios = []
            reference_kls = []
            for index in batch_indices.tolist():
                transition = transitions[index]
                logits, value = model.evaluate(
                    transition.observation,
                    transition.legal_action_representations,
                )
                value = value.squeeze()
                distribution = Categorical(logits=logits)
                chosen = torch.tensor(transition.chosen_action_index, device=model.device)
                new_log_probability = distribution.log_prob(chosen)
                ratio = torch.exp(new_log_probability - transition.old_log_probability)
                unclipped = ratio * advantages[index].to(model.device)
                clipped = torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                ) * advantages[index].to(model.device)
                policy_losses.append(-torch.minimum(unclipped, clipped))
                value_losses.append((value - returns[index].to(model.device)).pow(2))
                entropies.append(distribution.entropy())
                ratios.append(ratio.detach())
                if reference_log_probabilities is not None:
                    reference_log_prob = reference_log_probabilities[index].to(model.device)
                    current_log_prob = torch.log_softmax(logits, dim=0)
                    reference_probability = reference_log_prob.exp()
                    reference_kls.append(torch.sum(
                        reference_probability * (reference_log_prob - current_log_prob)
                    ))
            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()
            reference_kl = torch.stack(reference_kls).mean() if reference_kls else torch.zeros_like(entropy)
            loss = (
                policy_loss
                + value_loss_coefficient * value_loss
                - entropy_coefficient * entropy
                + reference_kl_coefficient * reference_kl
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            ratio_tensor = torch.stack(ratios)
            metrics.append((
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy.detach()),
                float(torch.mean(-torch.log(ratio_tensor.clamp_min(1e-8)))),
                float(((ratio_tensor - 1.0).abs() > clip_epsilon).float().mean()),
                float(reference_kl.detach()),
            ))
    count = len(metrics)
    return PPOUpdateMetrics(
        policy_loss=sum(item[0] for item in metrics) / count,
        value_loss=sum(item[1] for item in metrics) / count,
        entropy=sum(item[2] for item in metrics) / count,
        approx_kl=sum(item[3] for item in metrics) / count,
        clip_fraction=sum(item[4] for item in metrics) / count,
        reference_kl=sum(item[5] for item in metrics) / count,
        advantages_mean=float(advantages.mean()),
        transitions=len(transitions),
    )


__all__ = [
    "NeuralActorCritic",
    "PPOTrainingPlayer",
    "PPOUpdateMetrics",
    "RolloutResult",
    "RolloutTransition",
    "choose_opponent",
    "collect_rollout",
    "evaluate_greedy_model",
    "is_monotonic_evaluation_improvement",
    "weighted_evaluation_score",
    "gae_returns",
    "ppo_update",
    "terminal_reward",
]
