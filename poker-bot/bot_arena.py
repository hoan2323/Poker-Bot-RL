"""Common bot adapters and a safe bot-vs-bot match runner."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from environment import ShortDeckPokerEnv
from evaluate import heuristic_policy
from q_learning_agent import QLearningAgent


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_Q_TABLE = BASE_DIR / "q_table.npy"
CURRENT_Q_TABLE = BASE_DIR / "artifacts" / "current" / "q_table.npy"
STATE_SIZE = 186
STATE_KEY_SIZE = 15
ACTION_SIZE = 3
DEFAULT_NFSP_DIR = Path(r"C:\REL301m\code\poker\Đăng poker")
NFSP_MODEL_CANDIDATES = (
    "nfsp_agent_final.pt",
    "nfsp_agent2_final.pt",
    "checkpoints/nfsp_agent_final.pt",
    "player0_final.pt",
    "agent_final.pt",
)


class BotPolicy:
    """Common interface and display metadata for every arena participant."""

    name: str = "Bot"
    algorithm: str = "Unknown"
    model_type: str = "No model"
    model_path: str | Path | None = None
    source: str = "built-in"
    description: str = ""
    status: str = "Ready"

    def get_metadata(self) -> dict:
        """Return JSON-safe details used by setup, summary and replay views."""
        model_path = getattr(self, "model_path", None)
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "model_type": self.model_type,
            "model_path": str(model_path) if model_path else None,
            "source": self.source,
            "description": self.description,
            "status": self.status,
        }

    @property
    def is_ready(self) -> bool:
        return self.status in {"Ready", "Loaded"}

    def choose_action(self, observation, valid_actions, info=None) -> int:
        raise NotImplementedError


class RandomBotPolicy(BotPolicy):
    name = "Random Bot"
    algorithm = "Random Policy"
    model_type = "No trained model"
    source = "built-in"
    description = "Bot chọn ngẫu nhiên trong các action hợp lệ."
    status = "Ready"

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def choose_action(self, observation, valid_actions, info=None) -> int:
        if not valid_actions:
            raise ValueError("RandomBotPolicy received no valid actions")
        return self._rng.choice(list(valid_actions))


class HeuristicBotPolicy(BotPolicy):
    name = "Heuristic Bot"
    algorithm = "Rule-based Heuristic"
    model_type = "No trained model"
    source = "built-in"
    description = "Bot dùng luật thủ công dựa trên độ mạnh bài, không cần training."
    status = "Ready"

    def choose_action(self, observation, valid_actions, info=None) -> int:
        if not valid_actions:
            raise ValueError("HeuristicBotPolicy received no valid actions")
        action = heuristic_policy(observation, list(valid_actions))
        return action if action in valid_actions else list(valid_actions)[0]


def preferred_q_table_path():
    return CURRENT_Q_TABLE if CURRENT_Q_TABLE.is_file() else DEFAULT_Q_TABLE


def inspect_q_table(q_table_path):
    """Return compatibility diagnostics for the project's sparse dict Q-table."""
    path = Path(q_table_path).expanduser().resolve()
    result = {
        "path": str(path),
        "status": "Missing",
        "reason": "Không tìm thấy Q-table, hãy train model trước.",
        "numpy_shape": None,
        "logical_shape": None,
        "expected": f"sparse dict với state key dài {STATE_KEY_SIZE}, value shape ({ACTION_SIZE},)",
        "state_size": STATE_SIZE,
        "action_size": ACTION_SIZE,
        "metadata": None,
    }
    if not path.is_file():
        return result

    try:
        data = np.load(path, allow_pickle=True)
        result["numpy_shape"] = list(data.shape)
        table = data.item() if data.shape == () else data
    except Exception as exc:
        result.update(status="Incompatible", reason=f"Không đọc được file: {exc}")
        return result

    if not isinstance(table, dict):
        result.update(
            status="Incompatible",
            reason=f"Kiểu thực tế {type(table).__name__}; cần sparse dict Q-table.",
        )
        return result

    invalid_keys = [key for key in table if not isinstance(key, tuple) or len(key) != STATE_KEY_SIZE]
    invalid_values = [
        np.asarray(value).shape for value in table.values() if np.asarray(value).shape != (ACTION_SIZE,)
    ]
    result["logical_shape"] = [len(table), ACTION_SIZE]
    if invalid_keys or invalid_values:
        result.update(
            status="Incompatible",
            reason=(
                f"Shape logic thực tế {result['logical_shape']}, nhưng có "
                f"{len(invalid_keys)} state key hoặc {len(invalid_values)} action vector sai shape. "
                "Hãy train lại model."
            ),
        )
        return result

    metadata_path = path.parent / "training_metadata.json"
    if not metadata_path.is_file():
        result.update(
            status="Legacy compatible",
            reason="Loaded legacy Q-table without metadata.",
        )
        return result
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["metadata"] = metadata
    except (OSError, json.JSONDecodeError) as exc:
        result.update(status="Incompatible", reason=f"Metadata không đọc được: {exc}")
        return result

    expected_metadata = (
        metadata.get("format_version") == 3
        and metadata.get("state_size") == STATE_SIZE
        and metadata.get("action_size") == ACTION_SIZE
        and metadata.get("state_key_size") == STATE_KEY_SIZE
        and metadata.get("q_table_shape") == result["logical_shape"]
    )
    if not expected_metadata:
        result.update(
            status="Incompatible",
            reason=(
                "Metadata không tương thích: "
                f"format_version={metadata.get('format_version')}, "
                f"state_size={metadata.get('state_size')}, "
                f"action_size={metadata.get('action_size')}, "
                f"q_table_shape={metadata.get('q_table_shape')}. Hãy train lại model."
            ),
        )
        return result
    result.update(status="Compatible", reason="Q-table và metadata tương thích.")
    return result


class PokerBotRLPolicy(BotPolicy):
    name = "Poker-Bot-RL Bot"
    algorithm = "Tabular Q-Learning"
    model_type = "Q-table (.npy)"
    default_model_path = CURRENT_Q_TABLE
    fallback_model_path = DEFAULT_Q_TABLE
    source = r"C:\REL301m\code\poker\Poker-Bot-RL\poker-bot"
    description = "Bot dùng Q-Learning dạng bảng từ project Poker-Bot-RL."

    def __init__(self, q_table_path=None, strict=True):
        if q_table_path is None:
            q_table_path = preferred_q_table_path()
        self.model_path = Path(q_table_path).expanduser().resolve()
        self.q_table_path = self.model_path
        self.source = str(BASE_DIR)
        self.status = "Loading model"
        self.error_message = None
        self.agent = None
        self.diagnostics = inspect_q_table(self.q_table_path)
        self.warnings = []
        if self.diagnostics["status"] == "Missing":
            self.status = "Missing model"
            self.error_message = "Không tìm thấy Q-table, hãy train model trước."
            if strict:
                raise FileNotFoundError(self.error_message)
            return
        if self.diagnostics["status"] == "Incompatible":
            self.status = "Incompatible Q-table"
            self.error_message = (
                "Q-table không tương thích với environment/state_size hiện tại. "
                "Hãy train lại model hoặc chọn Heuristic/Random. "
                f"Chi tiết: {self.diagnostics['reason']} File: {self.q_table_path}"
            )
            if strict:
                raise RuntimeError(self.error_message)
            return
        if self.diagnostics["status"] == "Legacy compatible":
            self.warnings.append("Loaded legacy Q-table without metadata.")
        try:
            self.agent = QLearningAgent(action_size=3, epsilon=0.0)
            self.agent.load(self.q_table_path)
        except Exception as exc:
            self.status = "Load error"
            self.error_message = (
                f"Không load được Q-learning model '{self.q_table_path}': {exc}"
            )
            if strict:
                raise RuntimeError(self.error_message) from exc
            return
        self.status = "Loaded"

    def choose_action(self, observation, valid_actions, info=None) -> int:
        if not valid_actions:
            raise ValueError("PokerBotRLPolicy received no valid actions")
        if self.agent is None:
            raise RuntimeError(self.error_message or "Q-table chưa được load.")
        try:
            action = self.agent.choose_action(
                observation, list(valid_actions), opponent_profile=0
            )
        except Exception as exc:
            raise RuntimeError(f"Poker-Bot-RL policy không chọn được action: {exc}") from exc
        if action not in valid_actions:
            raise RuntimeError(f"Poker-Bot-RL policy trả action không hợp lệ: {action}")
        return action


def find_nfsp_model(code_dir=DEFAULT_NFSP_DIR):
    """Find a real Đăng checkpoint without ever substituting another policy."""
    code_dir = Path(code_dir).expanduser().resolve()
    for relative_path in NFSP_MODEL_CANDIDATES:
        candidate = code_dir / relative_path
        if candidate.is_file():
            return candidate
    if code_dir.is_dir():
        checkpoints = sorted(
            code_dir.rglob("*.pt"),
            key=lambda path: (len(path.relative_to(code_dir).parts), str(path).casefold()),
        )
        if checkpoints:
            return checkpoints[0].resolve()
    return None


def preferred_nfsp_model_path(code_dir=DEFAULT_NFSP_DIR):
    """Return the discovered checkpoint or the documented default missing path."""
    code_dir = Path(code_dir).expanduser().resolve()
    return find_nfsp_model(code_dir) or (code_dir / NFSP_MODEL_CANDIDATES[0])


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể tạo module spec cho {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DangPokerBotPolicy(BotPolicy):
    name = "Đăng poker Bot"
    algorithm = "Neural Fictitious Self-Play (NFSP)"
    model_type = "PyTorch checkpoint (.pt)"
    default_model_path = DEFAULT_NFSP_DIR / "nfsp_agent_final.pt"
    source = r"C:\REL301m\code\poker\Đăng poker"
    description = "Bot dùng NFSP/Deep RL từ folder Đăng poker."

    def __init__(self, model_path=None, code_dir=DEFAULT_NFSP_DIR, strict=True):
        self.code_dir = Path(code_dir).expanduser().resolve()
        self.source = str(self.code_dir)
        self.status = "Loading model"
        self.error_message = None
        self.network = None
        self.average_policy_network = None
        self._torch = None
        self.warnings = []
        if not self.code_dir.is_dir():
            self.model_path = (
                Path(model_path).expanduser().resolve()
                if model_path
                else self.code_dir / NFSP_MODEL_CANDIDATES[0]
            )
            self.status = "Import error"
            self.error_message = f"Không tìm thấy folder code NFSP: {self.code_dir}"
            if strict:
                raise FileNotFoundError(self.error_message)
            return

        if model_path:
            self.model_path = Path(model_path).expanduser().resolve()
        else:
            self.model_path = preferred_nfsp_model_path(self.code_dir)
        if not self.model_path.is_file():
            tried = ", ".join(NFSP_MODEL_CANDIDATES)
            self.status = "Missing model"
            self.error_message = (
                "Hiện chưa tìm thấy PyTorch checkpoint `.pt` cho Đăng poker Bot. "
                "Đăng poker Bot chưa thể đấu cho đến khi có model hợp lệ. "
                f"Folder: {self.code_dir}. Đã thử: {tried}"
            )
            if strict:
                raise FileNotFoundError(self.error_message)
            return

        try:
            import torch
        except ImportError as exc:
            self.status = "Import error"
            self.error_message = (
                "PyTorch chưa được cài. Cài torch riêng nếu muốn dùng NFSP model."
            )
            if strict:
                raise RuntimeError(self.error_message) from exc
            return

        config_path = self.code_dir / "config.py"
        networks_path = self.code_dir / "networks.py"
        if not config_path.is_file() or not networks_path.is_file():
            self.status = "Import error"
            self.error_message = "Folder NFSP thiếu config.py hoặc networks.py"
            if strict:
                raise FileNotFoundError(self.error_message)
            return

        suffix = uuid.uuid4().hex
        old_config = sys.modules.get("config")
        try:
            config_module = _load_module(config_path, f"_dang_config_{suffix}")
            sys.modules["config"] = config_module
            networks_module = _load_module(networks_path, f"_dang_networks_{suffix}")
        except Exception as exc:
            self.status = "Import error"
            self.error_message = f"Không import được NFSP network: {exc}"
            if strict:
                raise RuntimeError(self.error_message) from exc
            return
        finally:
            if old_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = old_config

        try:
            try:
                checkpoint = torch.load(
                    self.model_path, map_location="cpu", weights_only=True
                )
            except TypeError:
                checkpoint = torch.load(self.model_path, map_location="cpu")
            state_dict = (
                checkpoint.get("q_net", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            self.network = networks_module.QNetwork().to("cpu")
            self.network.load_state_dict(state_dict)
            self.network.eval()
            if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
                self.average_policy_network = networks_module.AveragePolicyNetwork().to(
                    "cpu"
                )
                self.average_policy_network.load_state_dict(checkpoint["policy_net"])
                self.average_policy_network.eval()
            else:
                self.warnings.append(
                    "Checkpoint không có average policy network; Arena dùng DQN best-response."
                )
            self._torch = torch
        except Exception as exc:
            self.status = "Load error"
            self.error_message = (
                f"Không load được NFSP model '{self.model_path}': {exc}"
            )
            if strict:
                raise RuntimeError(self.error_message) from exc
            return
        self.status = "Loaded"

    def choose_action(self, observation, valid_actions, info=None) -> int:
        if not valid_actions:
            raise ValueError("DangPokerBotPolicy received no valid actions")
        if self.network is None or self._torch is None:
            raise RuntimeError(self.error_message or "NFSP model chưa được load.")
        with self._torch.no_grad():
            state = self._torch.as_tensor(observation, dtype=self._torch.float32)
            q_values = self.network(state).squeeze(0).cpu().numpy()
        return max(valid_actions, key=lambda action: float(q_values[action]))


# Backward-compatible names for callers using the first Bot Arena implementation.
MainQBotPolicy = PokerBotRLPolicy
DangNFSPBotPolicy = DangPokerBotPolicy


def _safe_action(bot, observation, valid_actions, info, warnings):
    try:
        action = bot.choose_action(observation, valid_actions, info=info)
    except Exception as exc:
        raise RuntimeError(f"{bot.name} action lỗi: {exc}") from exc
    if action not in valid_actions:
        try:
            replacement = min(valid_actions, key=lambda candidate: abs(candidate - int(action)))
        except (TypeError, ValueError):
            replacement = valid_actions[0]
        warnings.append(
            f"{bot.name} trả action không hợp lệ {action}; Arena đã dùng action hợp lệ "
            f"gần nhất {replacement}. Policy và metadata của bot không bị thay đổi."
        )
        return int(replacement)
    return int(action)


def run_bot_match(
    bot_a,
    bot_b,
    num_games=1000,
    seed=42,
    max_steps_per_game=200,
    alternate_positions=False,
    progress_callback: Callable[[int, int], None] | None = None,
    replay_limit=5,
):
    """Run both bots on main's environment and return JSON-safe metrics."""
    if num_games < 1:
        raise ValueError("num_games must be at least 1")
    if max_steps_per_game < 1:
        raise ValueError("max_steps_per_game must be at least 1")
    if not bot_a.is_ready:
        raise RuntimeError(
            f"Bot A ({bot_a.name}) chưa sẵn sàng: "
            f"{getattr(bot_a, 'error_message', bot_a.status)}"
        )
    if not bot_b.is_ready:
        raise RuntimeError(
            f"Bot B ({bot_b.name}) chưa sẵn sàng: "
            f"{getattr(bot_b, 'error_message', bot_b.status)}"
        )

    random.seed(seed)
    np.random.seed(seed)
    env = ShortDeckPokerEnv()
    bot_a_metadata = bot_a.get_metadata()
    bot_b_metadata = bot_b.get_metadata()
    wins = {"a": 0, "b": 0}
    draws = timeouts = folds = showdowns = 0
    rewards_a = []
    pots = []
    steps_per_game = []
    action_counts = {"bot_a": Counter(), "bot_b": Counter()}
    replays = []
    warnings = list(getattr(bot_a, "warnings", [])) + list(getattr(bot_b, "warnings", []))

    for game_index in range(num_games):
        a_seat = game_index % 2 if alternate_positions else 0
        b_seat = 1 - a_seat
        env.reset(starting_player=game_index % 2)
        history = []
        steps = 0
        timed_out = False

        while not env.done:
            if steps >= max_steps_per_game:
                timed_out = True
                timeouts += 1
                break
            player = env.current_player
            bot_key, bot = ("bot_a", bot_a) if player == a_seat else ("bot_b", bot_b)
            valid_actions = list(env.get_valid_actions())
            observation = env.get_state(player)
            info = env._get_info() if hasattr(env, "_get_info") else {}
            action = _safe_action(bot, observation, valid_actions, info, warnings)
            action_counts[bot_key][action] += 1
            history.append(
                {"step": steps + 1, "player": player, "bot": bot_key, "action": action}
            )
            env.step(action)
            steps += 1

        winner = None if timed_out else env.winner
        if winner == a_seat:
            wins["a"] += 1
        elif winner == b_seat:
            wins["b"] += 1
        else:
            draws += 1

        if not timed_out:
            if env.round == 4:
                showdowns += 1
            else:
                folds += 1

        reward_a = 0.0 if timed_out else float(env.get_reward(a_seat))
        rewards_a.append(reward_a)
        pots.append(float(env.pot))
        steps_per_game.append(steps)
        replay = {
            "game": game_index + 1,
            "bot_a": f"{bot_a.name} - {bot_a.algorithm}",
            "bot_b": f"{bot_b.name} - {bot_b.algorithm}",
            "bot_a_algorithm": bot_a.algorithm,
            "bot_b_algorithm": bot_b.algorithm,
            "model_a": bot_a_metadata["model_path"] or bot_a_metadata["model_type"],
            "model_b": bot_b_metadata["model_path"] or bot_b_metadata["model_type"],
            "bot_a_seat": a_seat,
            "bot_b_seat": b_seat,
            "hole_cards": {"0": list(env.hands.get(0, [])), "1": list(env.hands.get(1, []))},
            "community_cards": list(env.board),
            "action_history": history,
            "actions": history,
            "winner": winner,
            "winner_bot": (
                bot_a.name
                if winner == a_seat
                else bot_b.name
                if winner == b_seat
                else "Draw / Timeout"
            ),
            "reward_a": reward_a,
            "reward_b": -reward_a,
            "pot": env.pot,
            "steps": steps,
            "timeout": timed_out,
        }
        replays.append(replay)
        replays = replays[-replay_limit:]
        if progress_callback:
            progress_callback(game_index + 1, num_games)

    return {
        "games": num_games,
        "bot_a_name": bot_a.name,
        "bot_b_name": bot_b.name,
        "bot_a_metadata": bot_a_metadata,
        "bot_b_metadata": bot_b_metadata,
        "summary": {
            "bot_a_algorithm": bot_a.algorithm,
            "bot_b_algorithm": bot_b.algorithm,
            "bot_a_used": f"{bot_a.name} - {bot_a.algorithm}",
            "bot_b_used": f"{bot_b.name} - {bot_b.algorithm}",
            "model_a": bot_a_metadata["model_path"] or bot_a_metadata["model_type"],
            "model_b": bot_b_metadata["model_path"] or bot_b_metadata["model_type"],
        },
        "bot_a_wins": wins["a"],
        "bot_b_wins": wins["b"],
        "draws": draws,
        "timeouts": timeouts,
        "bot_a_win_rate": wins["a"] / num_games,
        "bot_b_win_rate": wins["b"] / num_games,
        "draw_rate": draws / num_games,
        "avg_reward_a": float(np.mean(rewards_a)),
        "avg_reward_b": float(-np.mean(rewards_a)),
        "avg_pot": float(np.mean(pots)),
        "avg_steps": float(np.mean(steps_per_game)),
        "action_counts": {
            key: {str(action): counts.get(action, 0) for action in (0, 1, 2)}
            for key, counts in action_counts.items()
        },
        "fold_count": folds,
        "showdown_count": showdowns,
        "alternate_positions": bool(alternate_positions),
        "warnings": list(
            dict.fromkeys(
                warnings
                + list(getattr(bot_a, "warnings", []))
                + list(getattr(bot_b, "warnings", []))
            )
        )[:20],
        "replays": replays,
    }
