from environment import ShortDeckPokerEnv
from evaluate import (
    always_bet_policy,
    call_station_policy,
    evaluate_matchup,
    heuristic_policy,
    observation_for_player,
    random_policy,
    run_episode,
    summarize_results,
)


def always_check_policy(state, valid_actions):
    return 0


def test_observation_for_player_contains_only_that_players_hole_cards():
    env = ShortDeckPokerEnv()
    env.reset()
    state = observation_for_player(env, 1)
    assert {index for index, value in enumerate(state[:20]) if value} == set(env.hands[1])


def test_call_station_never_folds_when_call_is_valid():
    assert call_station_policy(None, [0, 1, 2]) == 0


def test_policies_return_valid_actions():
    env = ShortDeckPokerEnv()
    state = env.reset()
    for policy in (random_policy, heuristic_policy, always_bet_policy):
        action = policy(state, [0, 1])
        assert action in [0, 1]


def test_random_vs_random_episode_completes():
    result = run_episode(ShortDeckPokerEnv(), random_policy, random_policy)
    assert result.winner in (0, 1, None)
    assert result.end_reason in ("fold", "showdown")
    assert result.pot >= 2


def test_showdown_episode_uses_normalized_terminal_reward():
    result = run_episode(
        ShortDeckPokerEnv(), always_check_policy, always_check_policy
    )
    assert result.end_reason == "showdown"
    assert result.reward in (-result.pot, 0, result.pot)
    if result.winner == 0:
        assert result.reward == result.pot
    elif result.winner == 1:
        assert result.reward == -result.pot
    else:
        assert result.reward == 0


def test_summary_counts_all_showdown_outcomes():
    results = [
        run_episode(ShortDeckPokerEnv(), always_check_policy, always_check_policy)
        for _ in range(10)
    ]
    summary = summarize_results(results)
    assert summary["showdown_ends"] == 10
    assert (
        summary["showdown_wins"]
        + summary["showdown_losses"]
        + summary["showdown_draws"]
        == 10
    )


def test_evaluate_matchup_tracks_every_game():
    summary = evaluate_matchup(
        "always_bet_vs_random",
        always_bet_policy,
        random_policy,
        ShortDeckPokerEnv(),
        games=30,
    )
    assert summary["games"] == 30
    assert summary["by_position"][0]["games"] + summary["by_position"][1]["games"] == 30
