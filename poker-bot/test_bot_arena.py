from pathlib import Path

import numpy as np
import pytest

from bot_arena import (
    CURRENT_Q_TABLE,
    DEFAULT_Q_TABLE,
    DangPokerBotPolicy,
    HeuristicBotPolicy,
    PokerBotRLPolicy,
    RandomBotPolicy,
    find_nfsp_model,
    preferred_q_table_path,
    run_bot_match,
)
from bot_arena_ui import BOT_OPTIONS, DEFAULT_BOT_A, DEFAULT_BOT_B, _build_bot
from q_learning_agent import QLearningAgent
from training_artifacts import write_training_metadata


def sample_observation():
    state = np.zeros(186, dtype=np.float32)
    state[[0, 4]] = 1
    return state


def test_random_bot_always_returns_valid_action():
    bot = RandomBotPolicy(seed=42)
    for valid in ([0, 1], [0, 2], [0, 1, 2]):
        for _ in range(30):
            assert bot.choose_action(sample_observation(), valid) in valid


def test_heuristic_bot_always_returns_valid_action():
    bot = HeuristicBotPolicy()
    for valid in ([0, 1], [0, 2], [0, 1, 2]):
        assert bot.choose_action(sample_observation(), valid) in valid


def test_builtin_bot_metadata_identifies_algorithms():
    assert RandomBotPolicy().get_metadata()["algorithm"] == "Random Policy"
    assert HeuristicBotPolicy().get_metadata()["algorithm"] == "Rule-based Heuristic"


def test_default_matchup_is_dang_vs_poker_bot_rl():
    assert DEFAULT_BOT_A == "Đăng poker Bot"
    assert DEFAULT_BOT_B == "Poker-Bot-RL Bot"
    assert BOT_OPTIONS[:2] == (DEFAULT_BOT_A, DEFAULT_BOT_B)


def test_nfsp_metadata_is_available_when_model_is_missing(tmp_path):
    source = tmp_path / "dang-source"
    source.mkdir()
    bot = DangPokerBotPolicy(code_dir=source, strict=False)
    metadata = bot.get_metadata()
    assert metadata["name"] == "Đăng poker Bot"
    assert metadata["algorithm"] == "Neural Fictitious Self-Play (NFSP)"
    assert metadata["model_type"] == "PyTorch checkpoint (.pt)"
    assert metadata["status"] == "Missing model"
    assert metadata["model_path"].endswith("nfsp_agent_final.pt")
    assert not isinstance(bot, RandomBotPolicy)
    assert "Random Policy" not in metadata.values()
    assert "No trained model" not in metadata.values()
    assert "built-in" not in metadata.values()


def test_nfsp_finder_uses_a_recursive_pt_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "player0_final.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assert find_nfsp_model(tmp_path) == checkpoint.resolve()


def test_factory_only_creates_random_policy_for_explicit_random_selection(tmp_path):
    missing_model = tmp_path / "missing.pt"
    dang = _build_bot(DEFAULT_BOT_A, "unused.npy", missing_model, tmp_path)
    q_bot = _build_bot(DEFAULT_BOT_B, preferred_q_table_path(), "unused.pt", tmp_path)
    heuristic = _build_bot("Heuristic Bot", "unused.npy", "unused.pt", tmp_path)
    random_bot = _build_bot("Random Bot", "unused.npy", "unused.pt", tmp_path)

    assert isinstance(dang, DangPokerBotPolicy)
    assert isinstance(q_bot, PokerBotRLPolicy)
    assert isinstance(heuristic, HeuristicBotPolicy)
    assert isinstance(random_bot, RandomBotPolicy)
    assert not isinstance(dang, RandomBotPolicy)


def test_main_q_bot_loads_when_current_artifact_exists():
    model_path = preferred_q_table_path()
    if not Path(model_path).is_file():
        pytest.skip("q_table.npy is optional")
    bot = PokerBotRLPolicy(model_path)
    assert model_path == CURRENT_Q_TABLE
    assert bot.get_metadata()["model_path"] == str(CURRENT_Q_TABLE)
    assert bot.choose_action(sample_observation(), [0, 1]) in [0, 1]


def test_random_match_has_complete_metrics_and_never_crashes():
    result = run_bot_match(
        RandomBotPolicy(seed=1),
        RandomBotPolicy(seed=2),
        num_games=10,
        seed=42,
        max_steps_per_game=16,
        alternate_positions=True,
    )
    required = {
        "bot_a_wins",
        "bot_b_wins",
        "draws",
        "avg_reward_a",
        "avg_reward_b",
        "action_counts",
    }
    assert required.issubset(result)
    assert result["games"] == 10
    assert result["bot_a_wins"] + result["bot_b_wins"] + result["draws"] == 10
    assert all(replay["steps"] <= 16 for replay in result["replays"])
    assert result["bot_a_metadata"]["algorithm"] == "Random Policy"
    assert result["bot_b_metadata"]["algorithm"] == "Random Policy"
    assert result["summary"]["bot_a_algorithm"] == "Random Policy"
    assert result["summary"]["bot_b_algorithm"] == "Random Policy"
    assert all("bot_a_algorithm" in replay for replay in result["replays"])


def test_tiny_step_limit_records_timeout_instead_of_crashing():
    result = run_bot_match(
        RandomBotPolicy(seed=1),
        RandomBotPolicy(seed=2),
        num_games=3,
        max_steps_per_game=1,
    )
    assert result["timeouts"] >= 0
    assert result["bot_a_wins"] + result["bot_b_wins"] + result["draws"] == 3


def test_heuristic_vs_random_runs_ten_games():
    result = run_bot_match(
        HeuristicBotPolicy(), RandomBotPolicy(seed=4), num_games=10, seed=4
    )
    assert result["games"] == 10
    assert result["bot_a_wins"] + result["bot_b_wins"] + result["draws"] == 10


def test_main_q_missing_path_has_friendly_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Không tìm thấy Q-table"):
        PokerBotRLPolicy(tmp_path / "missing.npy")


def test_main_q_wrong_shape_has_friendly_error(tmp_path):
    path = tmp_path / "q_table.npy"
    np.save(path, np.zeros((5, 2)))
    with pytest.raises(RuntimeError, match="không tương thích"):
        PokerBotRLPolicy(path)


def test_main_q_loads_compatible_sparse_table_and_masks_actions(tmp_path):
    agent = QLearningAgent(action_size=3, epsilon=0.0)
    state = sample_observation()
    key = agent.get_state_key(state)
    agent.q_table[key] = np.array([1.0, 5.0, 100.0])
    path = tmp_path / "q_table.npy"
    agent.save(path)
    write_training_metadata(tmp_path, episodes=1, q_table_states=1)

    bot = PokerBotRLPolicy(path)
    assert bot.diagnostics["status"] == "Compatible"
    assert bot.get_metadata()["name"] == "Poker-Bot-RL Bot"
    assert bot.get_metadata()["algorithm"] == "Tabular Q-Learning"
    assert bot.get_metadata()["status"] == "Loaded"
    assert bot.choose_action(state, [0, 1]) == 1


def test_incompatible_q_table_keeps_metadata_in_safe_ui_mode(tmp_path):
    path = tmp_path / "q_table.npy"
    np.save(path, np.zeros((5, 2)))
    bot = PokerBotRLPolicy(path, strict=False)
    assert bot.get_metadata()["algorithm"] == "Tabular Q-Learning"
    assert bot.get_metadata()["status"] == "Incompatible Q-table"


def test_policy_error_stops_match_without_random_or_heuristic_fallback():
    class BrokenPolicy(RandomBotPolicy):
        name = "Broken selected bot"

        def choose_action(self, observation, valid_actions, info=None):
            raise RuntimeError("selected policy failed")

    with pytest.raises(RuntimeError, match="selected policy failed"):
        run_bot_match(BrokenPolicy(), RandomBotPolicy(seed=2), num_games=1)


def test_match_summary_records_selected_bot_names_and_algorithms():
    result = run_bot_match(
        HeuristicBotPolicy(), RandomBotPolicy(seed=2), num_games=2, seed=2
    )
    assert result["summary"]["bot_a_used"] == (
        "Heuristic Bot - Rule-based Heuristic"
    )
    assert result["summary"]["bot_b_used"] == "Random Bot - Random Policy"
