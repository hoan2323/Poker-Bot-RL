import numpy as np

from q_learning_agent import QLearningAgent
from opponent_model import OpponentModel
from train import shaping_reward


def sample_state():
    return np.array([0, 4, -1, -1, -1, -1, -1, 0, 0, 2, 0], dtype=np.int32)


def next_state():
    return np.array([0, 4, 10, 11, 12, -1, -1, 1, 0, 3, 0], dtype=np.int32)


def test_agent_initializes_q_table():
    agent = QLearningAgent(action_size=3)
    assert isinstance(agent.q_table, dict)
    assert agent.q_table == {}


def test_get_state_key_returns_tuple():
    agent = QLearningAgent(action_size=3)
    key = agent.get_state_key(sample_state())
    assert isinstance(key, tuple)
    assert len(key) == 15


def test_get_state_key_sets_hand_aware_flags():
    agent = QLearningAgent(action_size=3)
    # Ah Kh with flop Ac Qh Jh => one pair, top pair, flush draw, straight draw, pot bucket 1, round flop
    state = np.array([18, 14, 16, 10, 6, -1, -1, 1, 0, 4, 0], dtype=np.int32)
    key = agent.get_state_key(state)
    assert key == (1, 0, 1, 0, 0, 1, 1, 1, 0, 3, 1, 0, 0, 1, 0)


def test_get_state_key_sets_draw_and_risk_flags():
    agent = QLearningAgent(action_size=3)
    # Ah Kh with Qh Jh 10D => flush draw, broadway straight draw, large pot, active bet
    state = np.array([18, 14, 10, 6, 1, -1, -1, 1, 0, 7, 1], dtype=np.int32)
    key = agent.get_state_key(state)
    assert key == (4, 2, 0, 0, 0, 1, 1, 0, 0, 3, 3, 1, 0, 1, 0)


def test_get_state_key_sets_overpair():
    agent = QLearningAgent(action_size=3)
    # AA on KQJ board
    state = np.array([16, 17, 12, 8, 4, -1, -1, 1, 0, 6, 0], dtype=np.int32)
    key = agent.get_state_key(state)
    assert key[8] == 1
    assert key[10] == 2


def test_get_state_key_includes_opponent_profile():
    agent = QLearningAgent(action_size=3)
    key = agent.get_state_key(sample_state(), opponent_profile=OpponentModel.CALL_STATION)
    assert key[-1] == OpponentModel.CALL_STATION


def test_choose_action_only_chooses_valid_actions():
    agent = QLearningAgent(action_size=3, epsilon=1.0)
    state = sample_state()
    valid_actions = [0, 1]
    for _ in range(50):
        action = agent.choose_action(state, valid_actions)
        assert action in valid_actions


def test_choose_action_exploitation_only_uses_valid_actions():
    agent = QLearningAgent(action_size=3, epsilon=0.0)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 1.0
    agent.q_table[key][1] = 5.0
    agent.q_table[key][2] = 100.0
    action = agent.choose_action(state, [0])
    assert action == 0


def test_choose_action_randomizes_tied_best_q_values_when_enabled():
    agent = QLearningAgent(action_size=3, epsilon=0.0, random_tie_break=True)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 0.0
    agent.q_table[key][1] = 0.0
    agent.q_table[key][2] = -1.0

    actions = {agent.choose_action(state, [0, 1]) for _ in range(100)}

    assert actions == {0, 1}


def test_choose_action_uses_stable_tie_break_when_disabled():
    agent = QLearningAgent(action_size=3, epsilon=0.0, random_tie_break=False)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 0.0
    agent.q_table[key][1] = 0.0

    actions = {agent.choose_action(state, [0, 1]) for _ in range(100)}

    assert actions == {0}


def test_choose_action_calls_when_fold_q_not_above_margin():
    agent = QLearningAgent(action_size=3, epsilon=0.0, fold_margin=0.05)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 1.00
    agent.q_table[key][2] = 1.04

    action = agent.choose_action(state, [0, 2])

    assert action == 0


def test_choose_action_folds_when_fold_q_above_margin():
    agent = QLearningAgent(action_size=3, epsilon=0.0, fold_margin=0.05)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 1.00
    agent.q_table[key][2] = 1.06

    action = agent.choose_action(state, [0, 2])

    assert action == 2


def test_shaping_reward_does_not_reward_weak_fold():
    agent = QLearningAgent(action_size=3)
    state = np.array([0, 4, -1, -1, -1, -1, -1, 0, 0, 4, 1], dtype=np.int32)

    reward = shaping_reward(agent, state, 2, [0, 2])

    assert reward == 0.0


def test_get_state_key_includes_actions_this_round_count():
    agent = QLearningAgent(action_size=3)
    state = np.array([0, 4, -1, -1, -1, -1, -1, 0, 0, 2, 0, 1], dtype=np.int32)
    key = agent.get_state_key(state)

    assert key[12] == 1


def test_update_changes_q_value():
    agent = QLearningAgent(action_size=3, alpha=0.5, gamma=0.95, epsilon=0.0)
    state = sample_state()
    nxt = next_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    before = agent.q_table[key][1]
    agent.update(state, 1, 1.0, nxt, False, [0, 1])
    after = agent.q_table[key][1]
    assert after != before


def test_update_uses_only_valid_next_actions():
    agent = QLearningAgent(action_size=3, alpha=1.0, gamma=0.95, epsilon=0.0)
    state = sample_state()
    nxt = next_state()
    state_key = agent.get_state_key(state)
    next_key = agent.get_state_key(nxt)
    agent._ensure_state(state_key)
    agent._ensure_state(next_key)

    agent.q_table[next_key][0] = 1.0
    agent.q_table[next_key][1] = 10.0
    agent.q_table[next_key][2] = 100.0

    agent.update(state, 1, 1.0, nxt, False, [0, 1])

    expected = 1.0 + agent.gamma * 10.0
    assert agent.q_table[state_key][1] == expected


def test_update_uses_reward_only_when_no_valid_next_actions():
    agent = QLearningAgent(action_size=3, alpha=1.0, gamma=0.95, epsilon=0.0)
    state = sample_state()
    nxt = next_state()
    state_key = agent.get_state_key(state)
    agent._ensure_state(state_key)

    agent.update(state, 1, 2.5, nxt, False, [])

    assert agent.q_table[state_key][1] == 2.5


def test_update_uses_reward_only_when_done():
    agent = QLearningAgent(action_size=3, alpha=1.0, gamma=0.95, epsilon=0.0)
    state = sample_state()
    nxt = next_state()
    state_key = agent.get_state_key(state)
    next_key = agent.get_state_key(nxt)
    agent._ensure_state(state_key)
    agent._ensure_state(next_key)
    agent.q_table[next_key][0] = 999.0

    agent.update(state, 1, -3.0, nxt, True, [0])

    assert agent.q_table[state_key][1] == -3.0


def test_save_creates_q_table_npy(tmp_path):
    agent = QLearningAgent(action_size=3)
    state = sample_state()
    agent.choose_action(state, [0, 1])
    path = tmp_path / "q_table.npy"
    agent.save(path)
    assert path.exists()


def test_load_restores_q_table(tmp_path):
    agent = QLearningAgent(action_size=3)
    state = sample_state()
    key = agent.get_state_key(state)
    agent._ensure_state(key)
    agent.q_table[key][0] = 3.14
    agent.q_table[key][1] = -2.0

    path = tmp_path / "q_table.npy"
    agent.save(path)

    loaded = QLearningAgent(action_size=3)
    loaded.load(path)

    assert loaded.q_table.keys() == agent.q_table.keys()
    assert np.array_equal(loaded.q_table[key], agent.q_table[key])