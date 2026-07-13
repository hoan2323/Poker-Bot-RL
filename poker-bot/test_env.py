import numpy as np
import pytest

from environment import ShortDeckPokerEnv, compare_hands, evaluate_hand


def play_checks_to_showdown(env):
    state = None
    reward = 0
    done = False
    for _ in range(8):
        state, reward, done, _ = env.step(0)
        if done:
            break
    return state, reward, done


def test_reset_returns_player_zero_186_feature_state():
    env = ShortDeckPokerEnv()
    state = env.reset()

    assert state.shape == (186,)
    assert state.dtype == np.float32
    assert np.count_nonzero(state[:20]) == 2
    assert np.count_nonzero(state[80:100]) == 0
    assert env.pot == 2
    assert env.round == 0
    assert env.get_valid_actions() == [0, 1]


def test_player_state_hides_opponent_hole_cards():
    env = ShortDeckPokerEnv()
    env.reset()

    player_zero_state = env.get_state(0)
    player_one_state = env.get_state(1)
    assert set(np.flatnonzero(player_zero_state[:20])) == set(env.hands[0])
    assert set(np.flatnonzero(player_one_state[:20])) == set(env.hands[1])
    assert not set(env.hands[1]).issubset(set(np.flatnonzero(player_zero_state[:20])))


def test_checking_advances_all_four_betting_rounds():
    env = ShortDeckPokerEnv()
    env.reset()

    env.step(0)
    _, reward, done, _ = env.step(0)
    assert len(env.board) == 3
    assert env.round == 1
    assert reward == 0
    assert not done

    env.step(0)
    env.step(0)
    assert len(env.board) == 4
    assert env.round == 2

    env.step(0)
    env.step(0)
    assert len(env.board) == 5
    assert env.round == 3

    _, reward, done = play_checks_to_showdown(env)
    assert done
    assert env.round == 4
    assert reward in (-env.pot, 0, env.pot)


def test_raise_is_allowed_when_facing_a_bet():
    env = ShortDeckPokerEnv()
    env.reset()

    env.step(1)
    assert env.get_valid_actions() == [0, 1, 2]
    env.step(1)
    assert env.bet_size == 2
    assert env.current_player == 0
    assert env.pot == 4

    _, reward, done, _ = env.step(0)
    assert env.round == 1
    assert env.pot == 5
    assert reward == 0
    assert not done


def test_fold_awards_pot_to_the_other_player():
    env = ShortDeckPokerEnv()
    env.reset()
    env.step(1)

    _, reward, done, info = env.step(2)
    assert done
    assert info["winner"] == 0
    assert reward == env.pot
    with pytest.raises(RuntimeError):
        env.step(0)


def test_hand_evaluation_is_available_from_environment_module():
    four_of_a_kind = [0, 1, 2, 3, 4]
    one_pair = [4, 5, 8, 12, 17]

    assert evaluate_hand(four_of_a_kind)[0] > evaluate_hand(one_pair)[0]
    assert compare_hands(four_of_a_kind, one_pair) == 1
