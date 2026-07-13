from environment import ShortDeckPokerEnv
from game_results import normalized_step_info, resolve_terminal_result


def test_showdown_result_is_normalized_to_player_ids_and_player_zero_reward():
    env = ShortDeckPokerEnv()
    env.reset()
    done = False
    raw_info = None
    raw_reward = 0
    while not done:
        _, raw_reward, done, raw_info = env.step(0)

    reward, winner, end_reason = resolve_terminal_result(env, raw_reward)
    assert end_reason == "showdown"
    assert winner in (0, 1, None)
    assert reward in (-env.pot, 0, env.pot)
    if env.winner == 1:
        assert winner == 0
        assert reward == env.pot
    elif env.winner == -1:
        assert winner == 1
        assert reward == -env.pot
    else:
        assert winner is None
        assert reward == 0


def test_normalized_step_info_keeps_fold_winner_and_reward():
    env = ShortDeckPokerEnv()
    env.reset()
    env.step(1)
    _, raw_reward, done, info = env.step(2)

    reward, normalized_info = normalized_step_info(env, info, raw_reward)
    assert done
    assert reward == env.pot
    assert normalized_info["winner"] == 0
    assert normalized_info["end_reason"] == "fold"
