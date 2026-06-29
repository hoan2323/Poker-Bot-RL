import pytest
import numpy as np

from texas_holdenv import TexasHoldemEnv, SimplifiedTexasHoldemEnv, KuhnPokerEnv


def test_imports_successfully():
    env = TexasHoldemEnv()
    assert env is not None
    assert SimplifiedTexasHoldemEnv is TexasHoldemEnv
    assert KuhnPokerEnv is TexasHoldemEnv


def test_deck_has_20_unique_cards():
    env = TexasHoldemEnv()
    deck = env._create_deck()
    assert len(deck) == 20
    assert len(set(deck)) == 20
    assert set(env.ranks) == {10, 11, 12, 13, 14}
    assert set(env.suits) == {"C", "D", "H", "S"}


def test_reset_deals_2_cards_to_each_player():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    assert len(env.player_hands[0]) == 2
    assert len(env.player_hands[1]) == 2


def test_reset_hides_all_community_cards():
    env = TexasHoldemEnv()
    obs, info = env.reset(seed=123)
    assert env.community_cards == []
    assert list(obs[2:7]) == [-1, -1, -1, -1, -1]


def test_action_space_has_3_actions():
    env = TexasHoldemEnv()
    assert env.action_space.n == 3


def test_observation_shape_is_11():
    env = TexasHoldemEnv()
    obs, info = env.reset(seed=123)
    assert obs.shape == (11,)


def test_get_valid_actions_no_active_bet():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    assert env.get_valid_actions() == [0, 1]


def test_get_valid_actions_active_bet():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(1)
    assert env.get_valid_actions() == [0, 2]


def test_fold_invalid_when_no_active_bet():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    with pytest.raises(ValueError):
        env.step(2)


def test_bet_invalid_when_active_bet_exists():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(1)
    with pytest.raises(ValueError):
        env.step(1)


def test_check_check_preflop_reveals_3_flop_cards():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.round == "flop"
    assert len(env.community_cards) == 3
    assert list(obs[2:5]) != [-1, -1, -1]
    assert not terminated


def test_check_check_flop_reveals_1_turn_card():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(0)
    env.step(0)
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.round == "turn"
    assert len(env.community_cards) == 4
    assert obs[5] != -1
    assert not terminated


def test_check_check_turn_reveals_1_river_card():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.round == "river"
    assert len(env.community_cards) == 5
    assert obs[6] != -1
    assert not terminated


def test_check_check_river_triggers_showdown():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(0)
    assert env.round == "showdown"
    assert env.done is True
    assert terminated is True
    assert reward in [env.pot, -env.pot, 0]
    assert info["end_reason"] == "showdown"


def test_bet_call_preflop_reveals_flop():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(1)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.round == "flop"
    assert len(env.community_cards) == 3
    assert env.pot == 4
    assert not terminated


def test_fold_after_bet_ends_game():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(1)
    obs, reward, terminated, truncated, info = env.step(2)
    assert env.done is True
    assert terminated is True
    assert info["winner"] == 0
    assert info["end_reason"] == "fold"
    assert reward == env.pot


def test_reward_is_plus_pot_minus_pot_or_zero():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(0)
    assert reward in [env.pot, -env.pot, 0]


def test_reset_seed_is_reproducible():
    env1 = TexasHoldemEnv()
    env2 = TexasHoldemEnv()
    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)
    assert np.array_equal(obs1, obs2)
    assert env1.player_hands == env2.player_hands


def test_step_after_game_done_raises_runtime_error():
    env = TexasHoldemEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(2)
    with pytest.raises(RuntimeError):
        env.step(0)