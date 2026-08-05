from shards_ai.analysis.visited_state_benchmark import VisitedStateResult


def test_visited_metrics_track_rank_regret_and_divergence():
    result = VisitedStateResult()
    result.add_decision(
        phase="buy",
        neural_action_type="stop_buying",
        heuristic_action_type="buy_card",
        top1=False,
        rank=2,
        regret=1.5,
        heuristic_score=4.0,
    )
    result.add_decision(
        phase="buy",
        neural_action_type="buy_card",
        heuristic_action_type="buy_card",
        top1=True,
        rank=1,
        regret=0.0,
        heuristic_score=5.0,
    )

    metrics = result.by_phase["buy"].as_dict()
    assert metrics["records"] == 2
    assert metrics["top1_agreement"] == 0.5
    assert metrics["top3_agreement"] == 1.0
    assert metrics["mean_heuristic_rank"] == 1.5
    assert metrics["mean_heuristic_regret"] == 0.75
    assert metrics["divergence_rate"] == 0.5
