#!/usr/bin/env python3
"""Compare NeuralPlayer decisions with v008 on the states NeuralPlayer actually visits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer
from shards_ai.ai.action_representation import representation_for_neural_action
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.analysis.visited_state_benchmark import VisitedStateResult, write_html, write_json
from shards_ai.game import Game, GameRandom, GameRunner, PlayerId


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/visited_neural_vs_v008.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/analysis/visited_neural_vs_v008.html"))
    args = parser.parse_args()
    if args.games <= 0 or args.torch_threads <= 0:
        parser.error("games and torch-threads must be positive")
    if not args.checkpoint.exists() or not args.profile.exists():
        parser.error("checkpoint and heuristic profile must exist")
    torch.set_num_threads(args.torch_threads)
    scorer = NeuralPlayer.load_scorer(args.checkpoint)
    profile = load_profile(args.profile)
    result = VisitedStateResult()

    for game_index in range(args.games):
        seed = args.seed + game_index
        root_rng = GameRandom(seed)
        game = Game.new(seed=seed, rng=root_rng.derive("engine"))
        neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
        opponent_id = neural_id.opponent
        neural = NeuralPlayer(neural_id, None, root_rng.derive("neural"), scorer=scorer)
        opponent = HeuristicPlayer(
            opponent_id,
            profile.weights,
            profile.card_acquisition_weights,
            profile.constraint_weights,
        )
        counterfactual_heuristic = HeuristicPlayer(
            neural_id,
            profile.weights,
            profile.card_acquisition_weights,
            profile.constraint_weights,
        )
        runner = GameRunner(
            game,
            {neural_id: neural, opponent_id: opponent},
            max_actions=args.max_actions,
            max_turns=args.max_turns,
        )
        neural_decision_number = 0
        first_divergence: dict[str, object] | None = None

        def observe(observation, legal_actions, neural_action, player_id) -> None:
            nonlocal neural_decision_number, first_divergence
            if player_id is not neural_id:
                return
            neural_decision_number += 1
            before = game.state
            heuristic_action = counterfactual_heuristic.choose_action(before, legal_actions)
            representations = [representation_for_neural_action(action, observation) for action in legal_actions]
            with torch.inference_mode():
                scores = scorer(observation, representations).tolist()
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            neural_index = list(legal_actions).index(neural_action)
            heuristic_index = list(legal_actions).index(heuristic_action)
            heuristic_representation = representations[heuristic_index]
            rank = order.index(heuristic_index) + 1
            heuristic_score = counterfactual_heuristic.score_action(before, heuristic_action)
            regret = heuristic_score - counterfactual_heuristic.score_action(before, neural_action)
            neural_type = representations[neural_index].action_type
            heuristic_type = representations[heuristic_index].action_type
            result.add_decision(
                phase=observation.phase,
                neural_action_type=neural_type,
                heuristic_action_type=heuristic_type,
                legal_action_count=len(legal_actions),
                neural_card_id=representations[neural_index].card_definition_id,
                heuristic_card_id=heuristic_representation.card_definition_id,
                top1=neural_action == heuristic_action,
                rank=rank,
                regret=regret,
                heuristic_score=heuristic_score,
            )
            if first_divergence is None and neural_action != heuristic_action:
                first_divergence = {
                    "seed": seed,
                    "first_divergence_decision": neural_decision_number,
                    "turn_number": observation.turn_number,
                    "phase": observation.phase,
                    "neural_action": neural_type,
                    "heuristic_action": heuristic_type,
                    "heuristic_rank": rank,
                    "regret": regret,
                }

        runner.run(decision_observer=observe)
        result.games += 1
        result.first_divergence_by_game.append(first_divergence or {
            "seed": seed,
            "first_divergence_decision": None,
            "turn_number": None,
            "phase": None,
            "neural_action": None,
            "heuristic_action": None,
        })

    metadata = {
        "checkpoint": str(args.checkpoint), "profile": str(args.profile),
        "games": args.games, "seed": args.seed, "torch_threads": args.torch_threads,
    }
    write_json(result, args.output, metadata=metadata)
    write_html(result, args.html_output, metadata=metadata)
    print(json.dumps({"output": str(args.output), **result.overall.as_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
