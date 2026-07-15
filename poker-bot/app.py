from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    from environment import ShortDeckPokerEnv
    from q_learning_agent import QLearningAgent
    from evaluate import (
        call_station_policy,
        evaluate_matchup,
        heuristic_policy,
        load_trained_agent,
        mixed_opponent_policy,
        observation_for_player,
        q_policy,
        random_policy,
    )
    from game_results import normalized_step_info
    from training_artifacts import require_current_training_metadata
    from bot_arena_ui import render_bot_arena_tab
except Exception as exc:
    ShortDeckPokerEnv = None
    QLearningAgent = None
    call_station_policy = None
    evaluate_matchup = None
    heuristic_policy = None
    load_trained_agent = None
    mixed_opponent_policy = None
    observation_for_player = None
    q_policy = None
    random_policy = None
    normalized_step_info = None
    require_current_training_metadata = None
    render_bot_arena_tab = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
CURRENT_Q_TABLE_PATH = BASE_DIR / "artifacts" / "current" / "q_table.npy"
Q_TABLE_PATH = CURRENT_Q_TABLE_PATH if CURRENT_Q_TABLE_PATH.exists() else BASE_DIR / "q_table.npy"
TRAINING_FILES = {
    "Reward over time": BASE_DIR / "rewards.npy",
    "Win rate over time": BASE_DIR / "win_rates.npy",
    "Draw rate over time": BASE_DIR / "draw_rates.npy",
    "Loss rate over time": BASE_DIR / "loss_rates.npy",
}

ACTION_NAMES = {
    0: {"idle": "Check", "active": "Call"},
    1: {"idle": "Bet", "active": "Raise"},
    2: {"idle": "Fold", "active": "Fold"},
}


def format_card(card: Any) -> str:
    if card is None:
        return "-"

    if isinstance(card, str):
        return card

    if isinstance(card, (int, np.integer)) and 0 <= int(card) < 20:
        ranks = ["10", "J", "Q", "K", "A"]
        suits = ["\u2663", "\u2666", "\u2665", "\u2660"]
        return f"{ranks[int(card) // 4]}{suits[int(card) % 4]}"

    if not isinstance(card, tuple) or len(card) < 2:
        return str(card)

    rank, suit = card[0], card[1]
    rank_map = {10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
    suit_map = {
        "C": "\u2663",
        "D": "\u2666",
        "H": "\u2665",
        "S": "\u2660",
        "clubs": "\u2663",
        "diamonds": "\u2666",
        "hearts": "\u2665",
        "spades": "\u2660",
    }
    rank_text = rank_map.get(rank, str(rank).upper())
    suit_key = str(suit)
    suit_text = suit_map.get(suit_key, suit_map.get(suit_key.lower(), suit_key.upper()))
    return f"{rank_text}{suit_text}"


def format_cards(cards: Iterable[Any]) -> str:
    values = [format_card(card) for card in cards]
    return "  ".join(values) if values else "-"


def get_env_snapshot(env: Any) -> Dict[str, Any]:
    normalized_hands = dict(getattr(env, "hands", {0: [], 1: []}))
    community_cards = list(getattr(env, "board", []))
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
        "has_active_bet": bool(getattr(env, "bet_size", 0)),
        "bettor": None,
        "current_bet": getattr(env, "bet_size", 0),
        "stacks": getattr(env, "stacks", getattr(env, "chips", None)),
        "player_hands": normalized_hands,
        "community_cards": community_cards,
        "valid_actions": valid_actions,
        "done": bool(getattr(env, "done", False)),
    }


def _q_table_status() -> Dict[str, Any]:
    if not Q_TABLE_PATH.exists():
        return {"exists": False, "states": None, "error": None}

    try:
        data = np.load(Q_TABLE_PATH, allow_pickle=True)
        table = data.item() if getattr(data, "shape", None) == () else data
        states = len(table) if hasattr(table, "__len__") else None
        if require_current_training_metadata is not None:
            require_current_training_metadata(Q_TABLE_PATH.parent)
        return {"exists": True, "states": states, "error": None}
    except Exception as exc:
        return {"exists": True, "states": None, "error": str(exc)}


def _q_table_status_text() -> str:
    status = _q_table_status()
    if not status["exists"]:
        return "Q-table missing"
    if status["error"]:
        return "Q-table unreadable"
    states = status["states"] if status["states"] is not None else "unknown"
    return f"Q-table loaded - {states} states"


def _load_agent() -> Any:
    if QLearningAgent is None:
        return None

    if Q_TABLE_PATH.exists():
        try:
            agent = load_trained_agent(Q_TABLE_PATH)
            st.session_state.q_table_loaded = True
            st.session_state.q_table_error = None
            return agent
        except Exception as exc:
            st.session_state.q_table_loaded = False
            st.session_state.q_table_error = str(exc)
    else:
        st.session_state.q_table_loaded = False
        st.session_state.q_table_error = "q_table.npy not found"
    return QLearningAgent(action_size=3, epsilon=0.0)


def _action_label(action: int, env: Any) -> str:
    state = "active" if bool(getattr(env, "bet_size", 0)) else "idle"
    return ACTION_NAMES.get(action, {"idle": str(action), "active": str(action)}).get(state, str(action))


def _append_log(message: str) -> None:
    st.session_state.setdefault("game_log", [])
    st.session_state.game_log.append(message)


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
        env = ShortDeckPokerEnv()
        state = env.reset(starting_player=0)
        info = env._get_info()
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
        next_state, step_reward, done, info = env.step(action)
        reward, info = normalized_step_info(env, info, step_reward)
        st.session_state.last_state = next_state
        st.session_state.last_reward = reward
        st.session_state.last_info = info
        st.session_state.done = bool(done or getattr(env, "done", False))
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
                bot_state = env.get_state(bot_player)
            action = agent.choose_action(bot_state, valid_actions, opponent_profile=0)
        except Exception as exc:
            st.session_state.ui_error = f"Bot action failed: {exc}"
            _append_log(f"Bot (Player {bot_player}): action failed - {exc}")
            return

        _apply_step(action, f"Bot (Player {bot_player})")

    if safety_counter >= 20:
        st.session_state.ui_error = "Bot turn loop stopped after 20 steps."


def render_card_html(card: Any, hidden: bool = False) -> str:
    if hidden:
        return "<div class='card card-back'><span></span></div>"
    if card is None:
        return "<div class='card card-empty'><span>-</span></div>"

    text = format_card(card)
    if text.lower() in {"hidden", "back"}:
        return "<div class='card card-back'><span></span></div>"
    if text == "-":
        return "<div class='card card-empty'><span>-</span></div>"

    color_class = "red" if ("\u2665" in text or "\u2666" in text) else "black"
    return f"<div class='card {color_class}'><span>{escape(text)}</span></div>"


def _cards_html(cards: Iterable[Any], hidden: bool = False, size: int = 2) -> str:
    values = list(cards)
    if size and len(values) < size:
        values += [None] * (size - len(values))
    return "".join(render_card_html(card, hidden=hidden) for card in values)


def render_player_seat(
    label: str,
    badge: str,
    player_id: int,
    cards: Iterable[Any],
    active: bool,
    hidden: bool,
    position: str,
) -> str:
    active_class = "active" if active else ""
    return f"""
    <div class="player-seat seat-{position} {active_class}">
        <div class="player-avatar">{escape(badge[:1])}</div>
        <div class="player-copy">
            <div class="player-title">
                <span>{escape(label)}</span>
                <span class="player-badge">{escape(badge)}</span>
            </div>
            <div class="player-subtitle">Player {player_id}</div>
            <div class="hole-row">{_cards_html(cards, hidden=hidden, size=2)}</div>
        </div>
    </div>
    """


def render_poker_table(snapshot: Dict[str, Any], user_player: int, done: bool = False) -> None:
    bot_player = 1 - user_player
    current_player = snapshot["current_player"]
    community = list(snapshot["community_cards"])[:5]
    community += [None] * (5 - len(community))
    is_done = bool(done or snapshot["done"])
    bot_hidden = not is_done

    bot_seat = render_player_seat(
        "Q-Learning Bot",
        "BOT",
        bot_player,
        snapshot["player_hands"].get(bot_player, []),
        current_player == bot_player and not is_done,
        bot_hidden,
        "top",
    )
    human_seat = render_player_seat(
        "Human Player",
        "YOU",
        user_player,
        snapshot["player_hands"].get(user_player, []),
        current_player == user_player and not is_done,
        False,
        "bottom",
    )
    active_bet = "Active bet" if snapshot["has_active_bet"] else "No active bet"

    table_html = f"""
    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: transparent;
            color: #f7f4ea;
        }}

        .poker-stage {{
            width: 100%;
            height: 650px;
            padding: 18px;
            border-radius: 28px;
            background:
                radial-gradient(circle at 18% 0%, rgba(242, 201, 109, 0.18), transparent 28%),
                linear-gradient(135deg, rgba(7, 19, 17, 0.98), rgba(31, 18, 17, 0.96));
            overflow: hidden;
        }}

        .poker-table {{
            position: relative;
            width: 100%;
            height: 100%;
            border-radius: 48% / 36%;
            background:
                radial-gradient(ellipse at center, rgba(255, 255, 255, 0.10) 0%, transparent 58%),
                radial-gradient(ellipse at center, #178756 0%, #0c5d39 60%, #07341f 100%);
            border: 18px solid #53251f;
            box-shadow:
                inset 0 0 0 10px rgba(242, 201, 109, 0.22),
                inset 0 0 48px rgba(0, 0, 0, 0.42),
                0 24px 64px rgba(0, 0, 0, 0.36);
            overflow: hidden;
        }}

        .poker-table::before {{
            content: "";
            position: absolute;
            inset: 52px;
            border-radius: 48% / 36%;
            border: 1px dashed rgba(242, 201, 109, 0.30);
            pointer-events: none;
        }}

        .player-seat {{
            position: absolute;
            left: 50%;
            z-index: 3;
            width: min(480px, 78%);
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 14px 16px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(7, 17, 15, 0.86);
            box-shadow: 0 18px 35px rgba(0, 0, 0, 0.30);
        }}

        .player-seat.active {{
            border-color: rgba(242, 201, 109, 0.95);
            box-shadow: 0 0 0 4px rgba(242, 201, 109, 0.16), 0 18px 35px rgba(0, 0, 0, 0.30);
        }}

        .seat-top {{
            top: 24px;
        }}

        .seat-bottom {{
            bottom: 24px;
        }}

        .player-avatar {{
            width: 48px;
            aspect-ratio: 1;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: linear-gradient(135deg, #f2c96d, #ad5f3a);
            color: #24100c;
            font-weight: 900;
            flex: 0 0 auto;
        }}

        .player-copy {{
            flex: 1;
            min-width: 0;
        }}

        .player-title {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #f7f4ea;
            font-weight: 900;
        }}

        .player-subtitle {{
            margin: 2px 0 10px;
            color: rgba(247, 244, 234, 0.72);
            font-size: 12px;
        }}

        .player-badge {{
            border: 1px solid rgba(242, 201, 109, 0.34);
            border-radius: 999px;
            padding: 5px 10px;
            background: rgba(0, 0, 0, 0.22);
            color: #f2c96d;
            font-size: 11px;
            font-weight: 900;
        }}

        .table-center {{
            position: absolute;
            left: 8%;
            right: 8%;
            top: 228px;
            z-index: 2;
            display: grid;
            justify-items: center;
            gap: 14px;
        }}

        .pot-box {{
            display: grid;
            justify-items: center;
            min-width: 150px;
            padding: 12px 18px;
            border: 1px solid rgba(242, 201, 109, 0.56);
            border-radius: 16px;
            background: rgba(12, 17, 16, 0.82);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.30);
        }}

        .pot-box span {{
            color: rgba(247, 244, 234, 0.72);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .pot-box strong {{
            color: #f2c96d;
            font-size: 34px;
            line-height: 1;
        }}

        .community-row, .hole-row {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            flex-wrap: nowrap;
        }}

        .card {{
            width: 58px;
            aspect-ratio: 0.72;
            display: grid;
            place-items: center;
            border-radius: 10px;
            border: 1px solid rgba(0, 0, 0, 0.16);
            background: linear-gradient(145deg, #fffefa, #e8e4da);
            box-shadow: 0 11px 18px rgba(0, 0, 0, 0.28);
            font-weight: 900;
            font-size: 19px;
        }}

        .card.red {{
            color: #b9242c;
        }}

        .card.black {{
            color: #151515;
        }}

        .card-empty {{
            background: rgba(255, 255, 255, 0.10);
            color: rgba(255, 255, 255, 0.55);
            border: 1px dashed rgba(255, 255, 255, 0.26);
            box-shadow: none;
        }}

        .card-back {{
            background:
                linear-gradient(45deg, rgba(255,255,255,0.10) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.10) 75%),
                linear-gradient(45deg, rgba(255,255,255,0.10) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.10) 75%),
                linear-gradient(135deg, #7f1d28, #2b1018);
            background-size: 18px 18px;
            background-position: 0 0, 9px 9px, 0 0;
            border: 1px solid rgba(242, 201, 109, 0.36);
        }}

        .status-row {{
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .status-pill {{
            border: 1px solid rgba(242, 201, 109, 0.34);
            border-radius: 999px;
            padding: 7px 12px;
            background: rgba(0, 0, 0, 0.20);
            color: #f7f4ea;
            font-size: 13px;
            font-weight: 700;
        }}

        @media (max-width: 720px) {{
            .poker-stage {{
                height: 620px;
                padding: 10px;
            }}

            .poker-table {{
                border-width: 12px;
            }}

            .table-center {{
                left: 5%;
                right: 5%;
                top: 222px;
            }}

            .player-seat {{
                width: 90%;
            }}

            .card {{
                width: 48px;
                font-size: 16px;
            }}
        }}
    </style>
    <div class="poker-stage">
        <div class="poker-table">
            {bot_seat}
            <div class="table-center">
                <div class="pot-box">
                        <span>Total Pot</span>
                        <strong>{escape(str(snapshot["pot"]))}</strong>
                </div>
                <div class="community-row">
                    {_cards_html(community, size=5)}
                </div>
                <div class="status-row">
                    <span class="status-pill">Street: {escape(str(snapshot["phase"]).title())}</span>
                    <span class="status-pill">Current: Player {escape(str(current_player))}</span>
                    <span class="status-pill">{escape(active_bet)}</span>
                </div>
            </div>
            {human_seat}
        </div>
    </div>
    """
    components.html(table_html, height=650, scrolling=False)


def render_table(snapshot: Dict[str, Any], user_player: int) -> None:
    render_poker_table(snapshot, user_player, done=bool(snapshot["done"]))


def render_action_buttons(snapshot: Dict[str, Any], env: Any, user_player: int) -> None:
    st.markdown("<div class='section-title'>Controls</div>", unsafe_allow_html=True)
    if snapshot["done"]:
        if st.button("Play Next Hand", type="primary", use_container_width=True):
            init_game()
            st.rerun()
        return

    current_player = int(snapshot["current_player"])
    if current_player != user_player:
        st.info("Bot is thinking. The table will advance automatically.")
        return

    valid_actions = snapshot["valid_actions"]
    action_cols = st.columns(max(len(valid_actions), 1))
    for col, action in zip(action_cols, valid_actions):
        label = _action_label(action, env)
        help_text = {
            0: "Check when no bet is active, or call when facing a bet.",
            1: "Put one chip into the pot when no bet is active.",
            2: "Fold when facing an active bet.",
        }.get(action, None)
        if col.button(label, key=f"action_{action}", help=help_text, use_container_width=True):
            step_human_action(action)
            st.rerun()

    st.caption("Action mapping from env: 0 = Check/Call, 1 = Bet, 2 = Fold.")


def render_log_panel() -> None:
    log = st.session_state.get("game_log", [])
    items = "".join(f"<div class='log-item'><span>{idx}</span>{escape(item)}</div>" for idx, item in enumerate(log[-12:], start=max(1, len(log) - 11)))
    if not items:
        items = "<div class='log-empty'>No actions yet.</div>"
    st.markdown(
        f"""
        <div class="log-panel">
            <div class="panel-heading">Game Log</div>
            <div class="log-scroll">{items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_banner(snapshot: Dict[str, Any], user_player: int) -> None:
    if not snapshot["done"]:
        return

    info = st.session_state.get("last_info", {})
    reward = st.session_state.get("last_reward", 0)
    reward_label = "Reward" if user_player == 0 else "Env reward for Player 0"
    winner = info.get("winner") if isinstance(info, dict) else None

    if winner is None:
        css_class = "result-draw"
        title = "Draw"
        subtitle = f"{reward_label}: {reward}"
    elif int(winner) == user_player:
        css_class = "result-win"
        title = "You Win"
        subtitle = f"{reward_label}: {reward}"
    else:
        css_class = "result-loss"
        title = "Bot Wins"
        subtitle = f"{reward_label}: {reward}"

    st.markdown(
        f"""
        <div class="winner-banner {css_class}">
            <strong>{escape(title)}</strong>
            <span>{escape(subtitle)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    st.markdown(
        f"""
        <div class="hero-panel">
            <div>
                <div class="eyebrow">Texas Hold'em 2-player - Q-Learning Bot - Local Demo</div>
                <h1>Poker Bot RL - Live Table</h1>
                <p>Play against a Q-Learning poker bot on a short-deck Texas Hold'em table.</p>
            </div>
            <div class="hero-status">
                <span>{escape(_q_table_status_text())}</span>
                <strong>Human: Player {user_player}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("q_table_error") and not st.session_state.get("q_table_loaded", False):
        st.warning(f"Q-table not loaded: {st.session_state.q_table_error}")
    if st.session_state.get("ui_error"):
        st.error(st.session_state.ui_error)
    if user_player == 1:
        st.caption("Reward shown by the environment is from Player 0's perspective.")

    _render_result_banner(snapshot, user_player)

    table_col, side_col = st.columns([2.3, 1.0], gap="large")
    with table_col:
        render_poker_table(snapshot, user_player, done=bool(snapshot["done"]))
        render_action_buttons(snapshot, env, user_player)
    with side_col:
        render_log_panel()
        with st.container(border=True):
            st.markdown("**Hand Details**")
            st.write(f"Street: {snapshot['phase']}")
            st.write(f"Pot: {snapshot['pot']}")
            st.write(f"Current player: {snapshot['current_player']}")
            st.write(f"Active bet: {snapshot['has_active_bet']}")
            st.write(f"Done: {snapshot['done']}")
            st.write(f"Last reward: {st.session_state.get('last_reward', 0)}")

    with st.expander("Raw env snapshot"):
        st.json(
            {
                "phase": snapshot["phase"],
                "pot": snapshot["pot"],
                "current_player": snapshot["current_player"],
                "has_active_bet": snapshot["has_active_bet"],
                "bettor": snapshot["bettor"],
                "valid_actions": snapshot["valid_actions"],
                "done": snapshot["done"],
                "last_info": st.session_state.get("last_info", {}),
                "user_cards": format_cards(snapshot["player_hands"].get(user_player, [])),
                "bot_cards": format_cards(snapshot["player_hands"].get(1 - user_player, [])) if snapshot["done"] else "Hidden",
                "community_cards": format_cards(snapshot["community_cards"]),
            }
        )


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
    st.markdown(
        """
        <div class="page-card">
            <div class="eyebrow">Evaluation</div>
            <h2>Evaluate Bot</h2>
            <p>Run a quick local matchup using the existing Q-table. This does not train or overwrite model files.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if IMPORT_ERROR is not None:
        st.error(f"Could not import evaluation code: {IMPORT_ERROR}")
        return
    if not Q_TABLE_PATH.exists():
        st.warning("q_table.npy was not found. Train or provide a Q-table before evaluation.")
        return

    form_col, result_col = st.columns([1.0, 1.4], gap="large")
    with form_col:
        with st.container(border=True):
            game_option = st.radio("Games", ["100", "500", "1000", "Custom number"], horizontal=True)
            if game_option == "Custom number":
                games = int(st.number_input("Custom games", min_value=1, max_value=100_000, value=100, step=100))
            else:
                games = int(game_option)
            opponent_name = st.selectbox("Opponent", ["random", "call_station", "heuristic", "mixed"])

            if st.button("Run Evaluation", type="primary", use_container_width=True):
                opponent_policies = {
                    "random": random_policy,
                    "call_station": call_station_policy,
                    "heuristic": heuristic_policy,
                    "mixed": mixed_opponent_policy,
                }
                try:
                    env = ShortDeckPokerEnv()
                    agent = load_trained_agent(Q_TABLE_PATH)
                    with st.spinner("Running evaluation..."):
                        summary = evaluate_matchup(
                            f"q_vs_{opponent_name}",
                            q_policy(agent),
                            opponent_policies[opponent_name],
                            env,
                            games=games,
                        )
                    st.session_state.eval_summary = summary
                except Exception as exc:
                    st.session_state.eval_summary = None
                    st.error(f"Evaluation failed: {exc}")

    with result_col:
        summary = st.session_state.get("eval_summary")
        if not summary:
            st.info("Choose an opponent and run evaluation.")
            return

        row = _summary_to_row(summary)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Win Rate", f"{row['win_rate']:.4f}")
        metric_cols[1].metric("Draw Rate", f"{row['draw_rate']:.4f}")
        metric_cols[2].metric("Loss Rate", f"{row['loss_rate']:.4f}")
        metric_cols[3].metric("Avg Reward", f"{row['avg_reward']:.4f}")

        progress_cols = st.columns(3)
        progress_cols[0].caption("Wins")
        progress_cols[0].progress(min(max(row["win_rate"], 0.0), 1.0))
        progress_cols[1].caption("Draws")
        progress_cols[1].progress(min(max(row["draw_rate"], 0.0), 1.0))
        progress_cols[2].caption("Losses")
        progress_cols[2].progress(min(max(row["loss_rate"], 0.0), 1.0))
        st.dataframe([row], use_container_width=True, hide_index=True)

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
    st.markdown(
        """
        <div class="page-card">
            <div class="eyebrow">Training Dashboard</div>
            <h2>Training Results</h2>
            <p>Charts are loaded from existing .npy files. This page does not import plot.py or generate new training data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    curve_path = BASE_DIR / "training_curve.png"
    if curve_path.exists():
        with st.container(border=True):
            st.image(str(curve_path), caption="training_curve.png", use_container_width=True)
    else:
        st.info("Chua tim thay file training_curve.png")

    chart_items = list(TRAINING_FILES.items())
    for left_idx in range(0, len(chart_items), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, chart_items[left_idx : left_idx + 2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if not path.exists():
                        st.info(f"Chua tim thay file {path.name}")
                        continue
                    try:
                        data = np.load(path, allow_pickle=True)
                        data = np.asarray(data, dtype=np.float64)
                        sampled = _sample_series(data)
                        st.caption(f"{path.name}: {len(data):,} points")
                        st.line_chart(sampled)
                    except Exception as exc:
                        st.error(f"Could not read {path.name}: {exc}")


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --felt: #0f6a42;
            --felt-deep: #073822;
            --rail: #53251f;
            --gold: #f2c96d;
            --ink: #f7f4ea;
            --panel: rgba(10, 18, 18, 0.78);
            --muted: rgba(247, 244, 234, 0.72);
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(242, 201, 109, 0.16), transparent 28%),
                linear-gradient(135deg, #071311 0%, #10221d 44%, #2b1514 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1714 0%, #14251f 100%);
            border-right: 1px solid rgba(242, 201, 109, 0.18);
        }

        h1, h2, h3, p, label, .stMarkdown, .stRadio, .stSelectbox {
            color: var(--ink);
        }

        .hero-panel, .page-card {
            display: flex;
            justify-content: space-between;
            gap: 24px;
            align-items: center;
            padding: 24px 28px;
            margin: 8px 0 22px;
            border: 1px solid rgba(242, 201, 109, 0.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(14, 74, 49, 0.92), rgba(33, 24, 23, 0.92));
            box-shadow: 0 22px 50px rgba(0, 0, 0, 0.28);
        }

        .hero-panel h1, .page-card h2 {
            margin: 4px 0 8px;
            font-size: clamp(32px, 5vw, 54px);
            line-height: 1;
            letter-spacing: 0;
        }

        .page-card h2 {
            font-size: clamp(28px, 4vw, 40px);
        }

        .hero-panel p, .page-card p {
            margin: 0;
            color: var(--muted);
            font-size: 16px;
        }

        .eyebrow {
            color: var(--gold);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-status {
            min-width: 220px;
            display: grid;
            gap: 8px;
            justify-items: end;
        }

        .hero-status span, .hero-status strong, .table-pills span, .seat-badge {
            border: 1px solid rgba(242, 201, 109, 0.34);
            border-radius: 999px;
            padding: 7px 12px;
            background: rgba(0, 0, 0, 0.18);
        }

        .table-shell {
            padding: 18px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(242, 201, 109, 0.20), rgba(255, 255, 255, 0.04));
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
        }

        .poker-table {
            position: relative;
            min-height: 610px;
            border-radius: 46% / 34%;
            background:
                radial-gradient(ellipse at center, rgba(255, 255, 255, 0.08) 0%, transparent 58%),
                radial-gradient(ellipse at center, #178756 0%, #0c5d39 58%, #08351f 100%);
            border: 18px solid var(--rail);
            box-shadow:
                inset 0 0 0 10px rgba(242, 201, 109, 0.24),
                inset 0 0 48px rgba(0, 0, 0, 0.36),
                0 28px 70px rgba(0, 0, 0, 0.40);
            overflow: hidden;
        }

        .poker-table:before {
            content: "";
            position: absolute;
            inset: 54px;
            border-radius: 46% / 34%;
            border: 1px dashed rgba(242, 201, 109, 0.28);
        }

        .table-center {
            position: absolute;
            inset: 230px 70px auto 70px;
            z-index: 2;
            display: grid;
            justify-items: center;
            gap: 14px;
        }

        .pot-badge {
            display: grid;
            justify-items: center;
            min-width: 145px;
            padding: 12px 18px;
            border: 1px solid rgba(242, 201, 109, 0.52);
            border-radius: 16px;
            background: rgba(12, 17, 16, 0.78);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.28);
        }

        .pot-badge span {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
        }

        .pot-badge strong {
            color: var(--gold);
            font-size: 32px;
            line-height: 1;
        }

        .community-card-row, .hole-card-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            flex-wrap: nowrap;
        }

        .playing-card {
            width: 58px;
            aspect-ratio: 0.72;
            display: grid;
            place-items: center;
            border-radius: 10px;
            border: 1px solid rgba(0, 0, 0, 0.16);
            background: linear-gradient(145deg, #fffefa, #e8e4da);
            box-shadow: 0 11px 18px rgba(0, 0, 0, 0.28);
            font-weight: 900;
            font-size: 19px;
        }

        .card-red { color: #b9242c; }
        .card-black { color: #151515; }

        .card-empty {
            background: rgba(255, 255, 255, 0.10);
            color: rgba(255, 255, 255, 0.55);
            border: 1px dashed rgba(255, 255, 255, 0.26);
            box-shadow: none;
        }

        .card-back {
            background:
                linear-gradient(45deg, rgba(255,255,255,0.10) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.10) 75%),
                linear-gradient(45deg, rgba(255,255,255,0.10) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.10) 75%),
                linear-gradient(135deg, #7f1d28, #2b1018);
            background-size: 18px 18px;
            background-position: 0 0, 9px 9px, 0 0;
            border: 1px solid rgba(242, 201, 109, 0.36);
        }

        .table-pills {
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .table-pills span {
            color: var(--ink);
            font-size: 13px;
        }

        .table-mini {
            color: rgba(247, 244, 234, 0.70);
            font-size: 12px;
        }

        .seat {
            position: absolute;
            left: 50%;
            z-index: 3;
            width: min(470px, 78%);
            transform: translateX(-50%);
            display: flex;
            gap: 14px;
            align-items: center;
            padding: 14px 16px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(7, 17, 15, 0.82);
            box-shadow: 0 18px 35px rgba(0, 0, 0, 0.26);
        }

        .seat-top { top: 28px; }
        .seat-bottom { bottom: 28px; }

        .seat-active {
            border-color: rgba(242, 201, 109, 0.90);
            box-shadow: 0 0 0 4px rgba(242, 201, 109, 0.16), 0 18px 35px rgba(0, 0, 0, 0.28);
        }

        .seat-avatar {
            width: 48px;
            aspect-ratio: 1;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--gold), #ad5f3a);
            color: #24100c;
            font-weight: 900;
        }

        .seat-copy {
            flex: 1;
            min-width: 0;
        }

        .seat-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            font-weight: 900;
            color: var(--ink);
        }

        .seat-subtitle {
            color: var(--muted);
            font-size: 12px;
            margin: 2px 0 10px;
        }

        .seat-badge {
            color: var(--gold);
            font-size: 11px;
            font-weight: 900;
        }

        .section-title, .panel-heading {
            margin: 16px 0 10px;
            color: var(--gold);
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .log-panel, .details-card {
            border: 1px solid rgba(242, 201, 109, 0.18);
            border-radius: 18px;
            padding: 16px;
            background: var(--panel);
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.24);
        }

        .log-scroll {
            max-height: 390px;
            overflow-y: auto;
            display: grid;
            gap: 8px;
        }

        .log-item {
            display: grid;
            grid-template-columns: 28px 1fr;
            gap: 8px;
            align-items: center;
            padding: 9px 10px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.07);
            color: var(--ink);
            font-size: 14px;
        }

        .log-item span {
            display: grid;
            place-items: center;
            width: 24px;
            aspect-ratio: 1;
            border-radius: 50%;
            background: rgba(242, 201, 109, 0.20);
            color: var(--gold);
            font-size: 12px;
            font-weight: 900;
        }

        .log-empty {
            color: var(--muted);
            padding: 12px 0;
        }

        .winner-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            margin: 0 0 18px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.22);
        }

        .winner-banner strong {
            font-size: 24px;
        }

        .winner-banner span {
            color: rgba(255, 255, 255, 0.82);
        }

        .result-win { background: linear-gradient(135deg, #137a49, #104b33); }
        .result-loss { background: linear-gradient(135deg, #7f1d28, #3d1118); }
        .result-draw { background: linear-gradient(135deg, #7a6420, #3e3314); }

        .stButton > button {
            min-height: 52px;
            border-radius: 14px;
            border: 1px solid rgba(242, 201, 109, 0.28);
            font-weight: 900;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(242, 201, 109, 0.16);
            border-radius: 14px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.06);
        }

        @media (max-width: 900px) {
            .hero-panel {
                display: grid;
            }
            .hero-status {
                justify-items: start;
            }
            .poker-table {
                min-height: 560px;
                border-width: 12px;
            }
            .table-center {
                inset: 218px 26px auto 26px;
            }
            .playing-card {
                width: 48px;
                font-size: 16px;
            }
            .seat {
                width: 88%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> str:
    st.sidebar.title("Poker Bot RL")
    st.sidebar.caption("Texas Hold'em 2-player - Q-Learning Bot - Local Demo")
    mode = st.sidebar.radio(
        "Mode", ["Play vs Bot", "Evaluate Bot", "Training Results", "🤖 Bot Arena"]
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Q-table Status**")
    status = _q_table_status()
    if not status["exists"]:
        st.sidebar.warning("q_table.npy not found")
    elif status["error"]:
        st.sidebar.error(f"Found but unreadable: {status['error']}")
    else:
        states = status["states"] if status["states"] is not None else "unknown"
        st.sidebar.success(f"Available - {states} states")

    st.sidebar.divider()
    st.sidebar.markdown("**Game Controls**")
    selected_player = st.sidebar.radio("Human player", [0, 1], format_func=lambda value: f"Player {value}")
    if "user_player" not in st.session_state:
        st.session_state.user_player = selected_player
    elif int(st.session_state.user_player) != int(selected_player):
        st.session_state.user_player = selected_player
        init_game()

    if st.sidebar.button("New Hand", use_container_width=True):
        init_game()
        st.rerun()

    if st.sidebar.button("Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("**Theme Info**")
    st.sidebar.caption("Dark poker room, green felt table, visible pot, seats, cards, and action log.")
    return mode


def main() -> None:
    st.set_page_config(page_title="Poker Bot RL - Local Demo", layout="wide")
    inject_custom_css()

    if "user_player" not in st.session_state:
        st.session_state.user_player = 0

    mode = _render_sidebar()
    if mode == "Play vs Bot":
        render_play_page()
    elif mode == "Evaluate Bot":
        render_evaluate_page()
    elif mode == "Training Results":
        render_training_results_page()
    elif render_bot_arena_tab is not None:
        render_bot_arena_tab()
    else:
        st.error(f"Bot Arena không thể khởi tạo: {IMPORT_ERROR}")


if __name__ == "__main__":
    main()
