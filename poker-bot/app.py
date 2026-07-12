from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import streamlit as st

try:
    from texas_holdenv import TexasHoldemEnv
    from q_learning_agent import QLearningAgent
    from evaluate import (
        call_station_policy,
        evaluate_matchup,
        heuristic_policy,
        mixed_opponent_policy,
        observation_for_player,
        q_policy,
        random_policy,
    )
except Exception as exc:  # Streamlit should render a friendly error instead of a blank page.
    TexasHoldemEnv = None
    QLearningAgent = None
    call_station_policy = None
    evaluate_matchup = None
    heuristic_policy = None
    mixed_opponent_policy = None
    observation_for_player = None
    q_policy = None
    random_policy = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
Q_TABLE_PATH = BASE_DIR / "q_table.npy"
TRAINING_FILES = {
    "Reward over time": BASE_DIR / "rewards.npy",
    "Win rate over time": BASE_DIR / "win_rates.npy",
    "Draw rate over time": BASE_DIR / "draw_rates.npy",
    "Loss rate over time": BASE_DIR / "loss_rates.npy",
}

ACTION_NAMES = {
    0: {"idle": "Check", "active": "Call"},
    1: {"idle": "Bet", "active": "Bet"},
    2: {"idle": "Fold", "active": "Fold"},
}


def format_card(card: Any) -> str:
    if card is None:
        return "-"

    if isinstance(card, str):
        return card

    if not isinstance(card, tuple) or len(card) < 2:
        return str(card)

    rank, suit = card[0], card[1]
    rank_map = {10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
    suit_map = {
        "C": "♣",
        "D": "♦",
        "H": "♥",
        "S": "♠",
        "clubs": "♣",
        "diamonds": "♦",
        "hearts": "♥",
        "spades": "♠",
    }

    rank_text = rank_map.get(rank, str(rank).upper())
    suit_key = str(suit)
    suit_text = suit_map.get(suit_key, suit_map.get(suit_key.lower(), suit_key.upper()))
    return f"{rank_text}{suit_text}"


def format_cards(cards: Iterable[Any]) -> str:
    values = [format_card(card) for card in cards]
    return "  ".join(values) if values else "-"


def get_env_snapshot(env: Any) -> Dict[str, Any]:
    player_hands = getattr(env, "player_hands", {0: [], 1: []})
    if isinstance(player_hands, list):
        normalized_hands = {0: player_hands[0], 1: player_hands[1]}
    else:
        normalized_hands = {
            0: player_hands.get(0, []),
            1: player_hands.get(1, []),
        }

    community_cards = list(getattr(env, "community_cards", []))
    valid_actions: List[int] = []
    if not getattr(env, "done", False) and hasattr(env, "get_valid_actions"):
        try:
            valid_actions = list(env.get_valid_actions())
        except Exception:
            valid_actions = []

    return {
        "phase": getattr(env, "round", getattr(env, "phase", "unknown")),
        "pot": getattr(env, "pot", "-"),
        "current_player": getattr(env, "current_player", "-"),
        "has_active_bet": bool(getattr(env, "has_active_bet", False)),
        "bettor": getattr(env, "bettor", None),
        "current_bet": getattr(env, "current_bet", getattr(env, "bet", None)),
        "stacks": getattr(env, "stacks", getattr(env, "chips", None)),
        "player_hands": normalized_hands,
        "community_cards": community_cards,
        "valid_actions": valid_actions,
        "done": bool(getattr(env, "done", False)),
    }


def _action_label(action: int, env: Any) -> str:
    state = "active" if bool(getattr(env, "has_active_bet", False)) else "idle"
    return ACTION_NAMES.get(action, {"idle": str(action), "active": str(action)}).get(state, str(action))


def _append_log(message: str) -> None:
    st.session_state.setdefault("game_log", [])
    st.session_state.game_log.append(message)


def _q_table_status() -> Dict[str, Any]:
    if not Q_TABLE_PATH.exists():
        return {"exists": False, "states": None, "error": None}

    try:
        data = np.load(Q_TABLE_PATH, allow_pickle=True)
        table = data.item() if getattr(data, "shape", None) == () else data
        states = len(table) if hasattr(table, "__len__") else None
        return {"exists": True, "states": states, "error": None}
    except Exception as exc:
        return {"exists": True, "states": None, "error": str(exc)}


def _load_agent() -> Any:
    if QLearningAgent is None:
        return None

    action_size = 3
    try:
        env = TexasHoldemEnv() if TexasHoldemEnv is not None else None
        action_size = int(getattr(getattr(env, "action_space", None), "n", 3))
    except Exception:
        action_size = 3

    agent = QLearningAgent(action_size=action_size, epsilon=0.0)
    if Q_TABLE_PATH.exists():
        try:
            agent.load(Q_TABLE_PATH)
            agent.epsilon = 0.0
            st.session_state.q_table_loaded = True
            st.session_state.q_table_error = None
        except Exception as exc:
            st.session_state.q_table_loaded = False
            st.session_state.q_table_error = str(exc)
    else:
        st.session_state.q_table_loaded = False
        st.session_state.q_table_error = "q_table.npy not found"
    return agent


def init_game() -> None:
    if IMPORT_ERROR is not None:
        st.session_state.env = None
        st.session_state.agent = None
        st.session_state.done = True
        st.session_state.last_reward = 0
        st.session_state.last_info = {"error": str(IMPORT_ERROR)}
        st.session_state.game_log = [f"Import error: {IMPORT_ERROR}"]
        return

    try:
        env = TexasHoldemEnv()
        state, info = env.reset(options={"starting_player": 0})
        st.session_state.env = env
        st.session_state.agent = _load_agent()
        st.session_state.game_log = ["New hand started."]
        st.session_state.done = False
        st.session_state.last_reward = 0
        st.session_state.last_info = info
        st.session_state.last_state = state
        st.session_state.ui_error = None
    except Exception as exc:
        st.session_state.env = None
        st.session_state.agent = None
        st.session_state.done = True
        st.session_state.last_reward = 0
        st.session_state.last_info = {"error": str(exc)}
        st.session_state.game_log = [f"Could not start game: {exc}"]
        st.session_state.ui_error = str(exc)


def _apply_step(action: int, actor_label: str) -> None:
    env = st.session_state.get("env")
    if env is None:
        st.session_state.ui_error = "Environment is not available."
        return

    try:
        action_text = _action_label(action, env)
        next_state, reward, terminated, truncated, info = env.step(action)
        st.session_state.last_state = next_state
        st.session_state.last_reward = reward
        st.session_state.last_info = info
        st.session_state.done = bool(terminated or truncated or getattr(env, "done", False))
        _append_log(f"{actor_label}: {action_text}")
    except Exception as exc:
        st.session_state.ui_error = str(exc)
        _append_log(f"{actor_label}: step failed - {exc}")


def step_human_action(action: int) -> None:
    env = st.session_state.get("env")
    if env is None or st.session_state.get("done", False):
        return

    user_player = int(st.session_state.get("user_player", 0))
    if getattr(env, "current_player", None) != user_player:
        step_bot_until_human_turn_or_done()
        return

    _apply_step(action, f"You (Player {user_player})")
    if not st.session_state.get("done", False):
        step_bot_until_human_turn_or_done()


def step_bot_until_human_turn_or_done() -> None:
    env = st.session_state.get("env")
    agent = st.session_state.get("agent")
    if env is None or agent is None:
        return

    user_player = int(st.session_state.get("user_player", 0))
    safety_counter = 0

    while (
        not st.session_state.get("done", False)
        and getattr(env, "current_player", user_player) != user_player
        and safety_counter < 20
    ):
        safety_counter += 1
        bot_player = int(getattr(env, "current_player", 1 - user_player))
        try:
            valid_actions = env.get_valid_actions()
            if observation_for_player is not None:
                bot_state = observation_for_player(env, bot_player)
            else:
                bot_state = env._get_obs()
            action = agent.choose_action(bot_state, valid_actions, opponent_profile=0)
        except Exception as exc:
            st.session_state.ui_error = f"Bot action failed: {exc}"
            _append_log(f"Bot (Player {bot_player}): action failed - {exc}")
            return

        _apply_step(action, f"Bot (Player {bot_player})")

    if safety_counter >= 20:
        st.session_state.ui_error = "Bot turn loop stopped after 20 steps."


def _render_card_row(label: str, cards: Iterable[Any]) -> None:
    st.markdown(f"**{label}**")
    st.markdown(f"<div class='cards'>{format_cards(cards)}</div>", unsafe_allow_html=True)


def _render_game_log() -> None:
    st.markdown("**Hand Log**")
    log = st.session_state.get("game_log", [])
    if not log:
        st.caption("No actions yet.")
        return
    for index, item in enumerate(log, start=1):
        st.write(f"{index}. {item}")


def render_play_page() -> None:
    if st.session_state.get("env") is None:
        init_game()

    env = st.session_state.get("env")
    if env is None:
        st.error(st.session_state.get("ui_error") or "Environment could not be initialized.")
        return

    if not st.session_state.get("done", False) and getattr(env, "current_player", 0) != st.session_state.get("user_player", 0):
        step_bot_until_human_turn_or_done()

    snapshot = get_env_snapshot(env)
    user_player = int(st.session_state.get("user_player", 0))

    st.subheader("Play vs Bot")

    if st.session_state.get("q_table_error") and not st.session_state.get("q_table_loaded", False):
        st.warning(f"Q-table not loaded: {st.session_state.q_table_error}")

    if st.session_state.get("ui_error"):
        st.error(st.session_state.ui_error)

    cols = st.columns(4)
    cols[0].metric("Street", str(snapshot["phase"]).title())
    cols[1].metric("Pot", snapshot["pot"])
    cols[2].metric("Current Player", snapshot["current_player"])
    cols[3].metric("Active Bet", "Yes" if snapshot["has_active_bet"] else "No")

    detail_cols = st.columns(2)
    detail_cols[0].write(f"Bet current: {snapshot['current_bet'] if snapshot['current_bet'] is not None else 'Not exposed by env'}")
    detail_cols[1].write(f"Stacks/chips: {snapshot['stacks'] if snapshot['stacks'] is not None else 'Not exposed by env'}")

    left, right = st.columns(2)
    with left:
        _render_card_row(f"Your Cards - Player {user_player}", snapshot["player_hands"].get(user_player, []))
        other_player = 1 - user_player
        if snapshot["done"]:
            _render_card_row(f"Bot Cards - Player {other_player}", snapshot["player_hands"].get(other_player, []))
        else:
            _render_card_row(f"Bot Cards - Player {other_player}", ["Hidden", "Hidden"])

    with right:
        community = list(snapshot["community_cards"])[:5]
        community += [None] * (5 - len(community))
        _render_card_row("Community Cards", community)

    st.divider()

    if snapshot["done"]:
        info = st.session_state.get("last_info", {})
        reward = st.session_state.get("last_reward", 0)
        reward_label = "Reward" if user_player == 0 else "Env reward (Player 0)"
        winner = info.get("winner") if isinstance(info, dict) else None
        if winner is None:
            st.success("Result: Draw")
        elif int(winner) == user_player:
            st.success(f"Result: You win. {reward_label}: {reward}")
        else:
            st.error(f"Result: Bot wins. {reward_label}: {reward}")

        if st.button("New Hand", type="primary"):
            init_game()
            st.rerun()
    else:
        current_player = int(snapshot["current_player"])
        if current_player == user_player:
            st.markdown("**Your Action**")
            valid_actions = snapshot["valid_actions"]
            button_cols = st.columns(max(len(valid_actions), 1))
            for col, action in zip(button_cols, valid_actions):
                if col.button(_action_label(action, env), key=f"action_{action}", type="primary" if action == 0 else "secondary"):
                    step_human_action(action)
                    st.rerun()
        else:
            st.info("Bot is acting.")

    st.divider()
    _render_game_log()


def _summary_to_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    games = int(summary.get("games", 0) or 0)
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(float(summary.get("win_rate", 0.0)), 4),
        "draw_rate": round(draws / games if games else 0.0, 4),
        "loss_rate": round(losses / games if games else 0.0, 4),
        "avg_reward": round(float(summary.get("avg_reward", 0.0)), 4),
        "avg_pot": round(float(summary.get("avg_pot", 0.0)), 4),
    }


def render_evaluate_page() -> None:
    st.subheader("Evaluate Bot")

    if IMPORT_ERROR is not None:
        st.error(f"Could not import evaluation code: {IMPORT_ERROR}")
        return

    if not Q_TABLE_PATH.exists():
        st.warning("q_table.npy was not found. Train or provide a Q-table before evaluation.")
        return

    game_option = st.radio("Games", ["100", "500", "1000", "Custom number"], horizontal=True)
    if game_option == "Custom number":
        games = int(st.number_input("Custom games", min_value=1, max_value=100_000, value=100, step=100))
    else:
        games = int(game_option)

    opponent_name = st.selectbox("Opponent", ["random", "call_station", "heuristic", "mixed"])
    opponent_policies = {
        "random": random_policy,
        "call_station": call_station_policy,
        "heuristic": heuristic_policy,
        "mixed": mixed_opponent_policy,
    }

    if st.button("Run Evaluation", type="primary"):
        try:
            env = TexasHoldemEnv()
            agent = QLearningAgent(action_size=env.action_space.n, epsilon=0.0)
            agent.load(Q_TABLE_PATH)
            agent.epsilon = 0.0
            policy0 = q_policy(agent)
            policy1 = opponent_policies[opponent_name]
            with st.spinner("Running evaluation..."):
                summary = evaluate_matchup(f"q_vs_{opponent_name}", policy0, policy1, env, games=games)
            st.session_state.eval_summary = summary
        except Exception as exc:
            st.session_state.eval_summary = None
            st.error(f"Evaluation failed: {exc}")

    summary = st.session_state.get("eval_summary")
    if summary:
        row = _summary_to_row(summary)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Win Rate", f"{row['win_rate']:.4f}")
        metric_cols[1].metric("Draw Rate", f"{row['draw_rate']:.4f}")
        metric_cols[2].metric("Loss Rate", f"{row['loss_rate']:.4f}")
        metric_cols[3].metric("Average Reward", f"{row['avg_reward']:.4f}")
        st.table([row])

        with st.expander("Diagnostics"):
            st.json(
                {
                    "action_diagnostics": summary.get("action_diagnostics"),
                    "by_position": summary.get("by_position"),
                    "fold_rate_when_facing_bet": summary.get("fold_rate_when_facing_bet"),
                    "large_pot_win_rate": summary.get("large_pot_win_rate"),
                    "showdown_only_win_rate": summary.get("showdown_only_win_rate"),
                }
            )


def _sample_series(values: np.ndarray, max_points: int = 5000) -> np.ndarray:
    if values.size <= max_points:
        return values
    indexes = np.linspace(0, values.size - 1, max_points).astype(int)
    return values[indexes]


def render_training_results_page() -> None:
    st.subheader("Training Results")

    curve_path = BASE_DIR / "training_curve.png"
    if curve_path.exists():
        st.image(str(curve_path), caption="training_curve.png")
    else:
        st.info("Chưa tìm thấy file training_curve.png")

    for title, path in TRAINING_FILES.items():
        st.markdown(f"**{title}**")
        if not path.exists():
            st.info(f"Chưa tìm thấy file {path.name}")
            continue
        try:
            data = np.load(path, allow_pickle=True)
            data = np.asarray(data, dtype=np.float64)
            sampled = _sample_series(data)
            st.caption(f"{path.name}: {len(data):,} points")
            st.line_chart(sampled)
        except Exception as exc:
            st.error(f"Could not read {path.name}: {exc}")


def _render_sidebar() -> str:
    st.sidebar.title("Controls")
    mode = st.sidebar.radio("Mode", ["Play vs Bot", "Evaluate Bot", "Training Results"])

    if mode == "Play vs Bot":
        selected_player = st.sidebar.radio("You play as", [0, 1], format_func=lambda value: f"Player {value}")
        if "user_player" not in st.session_state:
            st.session_state.user_player = selected_player
        elif int(st.session_state.user_player) != int(selected_player):
            st.session_state.user_player = selected_player
            init_game()

    status = _q_table_status()
    st.sidebar.divider()
    st.sidebar.markdown("**Q-table Status**")
    if not status["exists"]:
        st.sidebar.warning("q_table.npy not found. Bot has no trained Q-table.")
    elif status["error"]:
        st.sidebar.error(f"q_table.npy found but could not be read: {status['error']}")
    else:
        states = status["states"] if status["states"] is not None else "unknown"
        st.sidebar.success(f"q_table.npy found. States: {states}")

    if st.sidebar.button("Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    return mode


def main() -> None:
    st.set_page_config(page_title="Poker Bot RL - Local Demo", layout="wide")
    st.markdown(
        """
        <style>
        .cards {
            border: 1px solid #d7dce2;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 56px;
            font-size: 28px;
            font-weight: 700;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Poker Bot RL - Local Demo")
    st.write("Project uses Q-Learning to play 2-player Texas Hold'em.")
    st.write("Play a hand against the bot, inspect the current hand state, or run a quick evaluation with the existing Q-table.")

    if "user_player" not in st.session_state:
        st.session_state.user_player = 0

    mode = _render_sidebar()

    if mode == "Play vs Bot":
        render_play_page()
    elif mode == "Evaluate Bot":
        render_evaluate_page()
    else:
        render_training_results_page()


if __name__ == "__main__":
    main()
