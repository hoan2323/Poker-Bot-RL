import pytest
from texas_holdenv import TexasHoldemEnv


def test_reset_runs():
    env = TexasHoldemEnv()
    state, info = env.reset()
    print("test_reset_runs:", state, info)
    assert state is not None
    assert isinstance(info, dict)


def test_action_space():
    env = TexasHoldemEnv()
    print("test_action_space:", env.action_space.n)
    assert env.action_space.n == 3


def test_deal_two_cards_each_player():
    env = TexasHoldemEnv()
    env.reset()
    print("test_deal_two_cards_each_player:", env.player_hands)
    assert len(env.player_hands[0]) == 2
    assert len(env.player_hands[1]) == 2


def test_no_duplicate_hole_cards():
    env = TexasHoldemEnv()
    env.reset()
    cards = env.player_hands[0] + env.player_hands[1]
    print("test_no_duplicate_hole_cards:", cards)
    assert len(cards) == len(set(cards))


def test_deck_has_52_unique_cards():
    env = TexasHoldemEnv()
    deck = env._create_deck()
    print("test_deck_has_52_unique_cards:", len(deck), len(set(deck)))

    assert len(deck) == 52
    assert len(deck) == len(set(deck))


def test_fold_ends_game():
    env = TexasHoldemEnv()
    env.reset()
    state, reward, terminated, truncated, info = env.step(2)
    print("test_fold_ends_game:", state, reward, terminated, truncated, info)

    assert terminated is True
    assert truncated is False
    assert reward in [-2, 2]
    assert info["winner"] in [0, 1]


def test_check_check_reveals_flop():
    env = TexasHoldemEnv()
    env.reset()

    env.step(0)
    state, reward, terminated, truncated, info = env.step(0)
    print("test_check_check_reveals_flop:", env.community_cards, state, reward, terminated, truncated, info)

    assert env.phase == "flop"
    assert len(env.community_cards) == 3
    assert terminated is False


def test_bet_increases_pot():
    env = TexasHoldemEnv()
    env.reset()

    old_pot = env.pot
    env.step(1)
    print("test_bet_increases_pot:", old_pot, env.pot)

    assert env.pot == old_pot + 1


def test_invalid_action_raises_error():
    env = TexasHoldemEnv()
    env.reset()
    print("test_invalid_action_raises_error: trying action 99")

    with pytest.raises(ValueError):
        env.step(99)


def test_step_after_done_raises_error():
    env = TexasHoldemEnv()
    env.reset()

    env.step(2)
    print("test_step_after_done_raises_error: game done =", env.done)

    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_seed_is_reproducible():
    env1 = TexasHoldemEnv()
    env2 = TexasHoldemEnv()

    state1, info1 = env1.reset(seed=123)
    state2, info2 = env2.reset(seed=123)
    print("test_reset_seed_is_reproducible:", state1.tolist(), state2.tolist(), env1.player_hands, env2.player_hands)

    assert state1.tolist() == state2.tolist()
    assert env1.player_hands == env2.player_hands


def test_showdown_after_flop_actions():
    env = TexasHoldemEnv()
    env.reset()

    env.step(0)
    env.step(0)

    env.step(0)
    state, reward, terminated, truncated, info = env.step(0)
    print("test_showdown_after_flop_actions:", state, reward, terminated, truncated, info)

    assert terminated is True
    assert truncated is False
    assert reward in [-2, 0, 2]
    assert info["winner"] in [0, 1, None]


def test_game_can_finish():
    env = TexasHoldemEnv()
    state, info = env.reset()

    terminated = False
    reward = 0

    for _ in range(10):
        if terminated:
            break

        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

    print("test_game_can_finish:", state, reward, terminated, truncated, info)
    assert terminated is True
    assert truncated is False
    assert reward in [-2, 0, 2]
