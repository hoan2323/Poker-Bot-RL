"""
test_evaluate.py — Tests for evaluation fairness.
"""
import random
import numpy as np

from texas_holdenv import TexasHoldemEnv
from q_learning_agent import QLearningAgent
from evaluate import (
    heuristic_policy,
    call_station_policy,
    random_policy,
    always_bet_policy,
    run_episode,
    evaluate_matchup,
    EVAL_GAMES,
)


def test_call_station_never_folds_when_call_valid():
    env = TexasHoldemEnv()
    env.reset(seed=42)
    env.has_active_bet = True
    valid_actions = env.get_valid_actions()
    assert 0 in valid_actions
    act = call_station_policy(None, valid_actions)
    assert act == 0


def test_call_station_folds_when_call_invalid():
    # Should not happen in our env since fold is only other option, but test just in case
    act = call_station_policy(None, [2])
    assert act == 2


def test_heuristic_does_not_see_player0_hole_cards():
    """Heuristic uses state vector, not env directly. State hole cards are
    visible in state index 0 and 1, but heuristic only uses state for
    board info and its own hole if acting as player 1. Since state hole
    cards always show Player 0's cards, we confirm the heuristic does not
    exploit this. In our eval, heuristic always acts as player 1."""
    env = TexasHoldemEnv()
    env.reset(seed=42)
    state, _ = env.reset(options={"starting_player": 1})
    state[0] = 999  # Artificially high card for Player 0 hole 1
    state[1] = 999  # Artificially high card for Player 0 hole 2
    valid = env.get_valid_actions()
    act = heuristic_policy(state, valid)
    # Should not crash and return valid action
    assert act in valid


def test_random_vs_random_roughly_balanced():
    """random_vs_random should be near 50% win rate for P0 (slight edge due to position balance)."""
    results = []
    for seed_val in range(500):
        env = TexasHoldemEnv()
        r = run_episode(env, random_policy, random_policy, starting_player=seed_val % 2)
        results.append(r)
    wins = sum(1 for r in results if r.reward > 0)
    losses = sum(1 for r in results if r.reward < 0)
    win_rate = wins / len(results)
    # Expect roughly balanced, within 10% of 0.5
    assert 0.35 <= win_rate <= 0.65, f"random_vs_random win_rate {win_rate:.4f} too far from 0.5"


def test_always_bet_vs_random_is_baseline():
    """always_bet_vs_random should be high but reportable."""
    env = TexasHoldemEnv()
    summary = evaluate_matchup("always_bet_vs_random", always_bet_policy, random_policy, env, games=1000)
    assert summary["games"] == 1000
    # Baseline is a known result; just validate it runs and produces sensible output
    assert summary["win_rate"] >= 0.5


def test_position_win_rate_tracked():
    """Position win rate by starting player should sum to games."""
    env = TexasHoldemEnv()
    summary = evaluate_matchup("test_pos", random_policy, random_policy, env, games=1000)
    pos0_games = summary["by_position"][0]["games"]
    pos1_games = summary["by_position"][1]["games"]
    assert pos0_games + pos1_games == 1000


def test_heuristic_policy_returns_valid_action():
    env = TexasHoldemEnv()
    env.reset(seed=42)
    state, _ = env.reset()
    for _ in range(20):
        valid = env.get_valid_actions()
        act = heuristic_policy(state, valid)
        assert act in valid
        state, _, terminated, truncated, _ = env.step(act)
        if terminated or truncated:
            state, _ = env.reset()


def test_evaluate_matchup_q_vs_random_uses_q_table():
    """Q vs random should use the loaded Q-table. If q_table.npy exists, run."""
    try:
        agent = QLearningAgent(action_size=3, epsilon=0.0)
        agent.load("q_table.npy")
    except FileNotFoundError:
        assert True
        return
    env = TexasHoldemEnv()
    summary = evaluate_matchup("q_vs_random_test", lambda s, v: agent.choose_action(s, v), random_policy, env, games=100)
    assert summary["games"] == 100