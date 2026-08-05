from dataclasses import asdict

import torch

from shards_ai.analysis.neural_imitation import analyze_records, decision_types_for
from shards_ai.game import Game, PlayerId


class FixedModel:
    def eval(self):
        return self

    def __call__(self, _observation, actions):
        return torch.tensor([float(index) for index in range(len(actions))])


def _record(*, phase="buy", action_type="buy_card", target=None):
    observation = asdict(Game.new(seed=700).neural_observation_for(PlayerId.PLAYER_1))
    observation["phase"] = phase
    representations = [
        {"action_type": "buy_card", "phase": phase, "river_slot": 0, "card_definition_id": "crystal"},
        {"action_type": "recruit_mercenary", "phase": phase, "river_slot": 1, "card_definition_id": "void_assassin"},
        {"action_type": "assign_power", "phase": phase, "target": target},
    ]
    chosen_index = {"buy_card": 0, "recruit_mercenary": 1, "assign_power": 2}[action_type]
    return {
        "observation": observation,
        "action_representations": representations,
        "heuristic_scores": [10.0, 8.0, 6.0],
        "chosen_action_index": chosen_index,
        "chosen_action": {"action_type": action_type},
    }


def test_analysis_reports_top1_top3_and_heuristic_regret():
    result = analyze_records(FixedModel(), [_record()])

    assert result.overall.as_dict() == {
        "records": 1,
        "top1_agreement": 0.0,
        "top3_agreement": 1.0,
        "mean_heuristic_score": 10.0,
        "mean_heuristic_regret": 4.0,
    }
    assert result.by_decision_type["Achat"].records == 1
    assert result.by_phase["buy"].records == 1


def test_targeting_is_an_overlapping_decision_family():
    assert decision_types_for({"action_type": "assign_power", "target": "opponent"}) == ("Attaque", "Ciblage")


def test_analysis_rejects_inconsistent_action_scores():
    record = _record()
    record["heuristic_scores"] = [10.0]

    try:
        analyze_records(FixedModel(), [record])
    except ValueError as error:
        assert "action/score lengths" in str(error)
    else:
        raise AssertionError("invalid action/score lengths must be rejected")
