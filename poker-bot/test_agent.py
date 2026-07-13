import numpy as np

from environment import ShortDeckPokerEnv
from opponent_model import OpponentModel
from q_learning_agent import QLearningAgent
from train import shaping_reward


def sample_state():
    env = ShortDeckPokerEnv()
    return env.reset()


def next_state():
    env = ShortDeckPokerEnv()
    env.reset()
    env.step(0)
    state, _, _, _ = env.step(0)
    return state


def test_get_state_key_accepts_new_186_feature_state():
    agent = QLearningAgent(action_size=3)
    key = agent.get_state_key(sample_state())

    assert isinstance(key, tuple)
    assert len(key) == 15


def test_get_state_key_includes_round_and_active_bet():
    env = ShortDeckPokerEnv()
    state = env.reset()
    env.step(1)
    state = env.get_state(env.current_player)
    key = QLearningAgent(action_size=3).get_state_key(state)

    assert key[11] == 1
    assert key[13] == 0


def test_get_state_key_includes_opponent_profile():
    key = QLearningAgent(action_size=3).get_state_key(
        sample_state(), opponent_profile=OpponentModel.CALL_STATION
    )
    assert key[-1] == OpponentModel.CALL_STATION


def test_choose_action_only_chooses_valid_actions():
    agent = QLearningAgent(action_size=3, epsilon=1.0)
    for _ in range(50):
        assert agent.choose_action(sample_state(), [0, 1]) in [0, 1]


def test_choose_action_exploitation_only_uses_valid_actions():
    agent = QLearningAgent(action_size=3, epsilon=0.0)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][:] = [1.0, 5.0, 100.0]
    assert agent.choose_action(state, [0]) == 0


def test_shaping_reward_handles_new_state():
    agent = QLearningAgent(action_size=3)
    reward = shaping_reward(agent, sample_state(), 0, [0, 1])
    assert isinstance(reward, float)


def test_update_uses_only_valid_next_actions():
    agent = QLearningAgent(action_size=3, alpha=1.0, gamma=0.95, epsilon=0.0)
    state = sample_state()
    nxt = next_state()
    state_key = agent.get_state_key(state)
    next_key = agent.get_state_key(nxt)
    agent._ensure_state(state_key)
    agent._ensure_state(next_key)
    agent.q_table[next_key][:] = [1.0, 10.0, 100.0]

    agent.update(state, 1, 1.0, nxt, False, [0, 1])
    assert agent.q_table[state_key][1] == 1.0 + agent.gamma * 10.0


def test_save_and_load_restores_q_table(tmp_path):
    agent = QLearningAgent(action_size=3)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 3.14
    path = tmp_path / "q_table.npy"
    agent.save(path)

    loaded = QLearningAgent(action_size=3)
    loaded.load(path)
    assert np.array_equal(loaded.q_table[key], agent.q_table[key])
