from shards_ai.game import GameRunner


def test_transition_observer_receives_detached_state_by_default() -> None:
    runner = GameRunner.random_duel(seed=101, max_turns=1)
    observed = []

    runner.run(transition_observer=lambda before, action, after, player_id: observed.append(after))

    assert observed
    assert observed[0] is not runner.game.state


def test_trusted_transition_observer_can_receive_live_post_state() -> None:
    runner = GameRunner.random_duel(seed=102, max_turns=1)
    observed = []

    runner.run(
        transition_observer=lambda before, action, after, player_id: observed.append(after),
        observer_receives_detached_state=False,
    )

    assert observed
    assert observed[0] is runner.game.state


def test_transition_observer_can_use_a_separate_before_state_factory() -> None:
    runner = GameRunner.random_duel(seed=106, max_turns=1)
    observed = []

    runner.run(
        transition_observer=lambda before, action, after, player_id: observed.append(
            (before, after)
        ),
        observer_receives_detached_state=False,
        players_receive_detached_observation=False,
        observer_before_state_factory=runner.game.shaping_observation_for,
    )

    assert observed
    before, after = observed[0]
    assert before is not runner.game.state
    assert before.players[runner.game.active_player].hand == []
    assert after is runner.game.state


def test_read_only_players_receive_live_observation_by_default() -> None:
    runner = GameRunner.random_duel(seed=103, max_turns=1)
    observed = []
    player = runner.players[runner.game.active_player]
    original_choose_action = player.choose_action

    def choose_action(observation, legal_actions):
        observed.append(observation)
        return original_choose_action(observation, legal_actions)

    player.choose_action = choose_action
    runner.run()

    assert observed
    assert observed[0] is runner.game.state


def test_explicit_detached_player_observation_remains_available() -> None:
    runner = GameRunner.random_duel(seed=105, max_turns=1)
    observed = []
    player = runner.players[runner.game.active_player]
    original_choose_action = player.choose_action

    def choose_action(observation, legal_actions):
        observed.append(observation)
        return original_choose_action(observation, legal_actions)

    player.choose_action = choose_action
    runner.run(players_receive_detached_observation=True)

    assert observed
    assert observed[0] is not runner.game.state


def test_trusted_player_can_receive_live_observation() -> None:
    runner = GameRunner.random_duel(seed=104, max_turns=1)
    observed = []
    player = runner.players[runner.game.active_player]
    original_choose_action = player.choose_action

    def choose_action(observation, legal_actions):
        observed.append(observation)
        return original_choose_action(observation, legal_actions)

    player.choose_action = choose_action
    runner.run(players_receive_detached_observation=False)

    assert observed
    assert observed[0] is runner.game.state


def test_decision_observer_receives_legal_actions_before_transition() -> None:
    runner = GameRunner.random_duel(seed=107, max_turns=1)
    observed = []

    runner.run(
        decision_observer=lambda before, actions, chosen, player_id: observed.append(
            (before.phase, tuple(actions), chosen, player_id)
        )
    )

    assert observed
    assert observed[0][2] in observed[0][1]
