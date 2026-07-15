"""Compact Streamlit UI for Đăng poker Bot vs Poker-Bot-RL Bot."""

import json

import streamlit as st

from bot_arena import (
    DEFAULT_NFSP_DIR,
    DangPokerBotPolicy,
    HeuristicBotPolicy,
    PokerBotRLPolicy,
    RandomBotPolicy,
    preferred_nfsp_model_path,
    preferred_q_table_path,
    run_bot_match,
)


DEFAULT_BOT_A = "Đăng poker Bot"
DEFAULT_BOT_B = "Poker-Bot-RL Bot"
BOT_OPTIONS = (
    DEFAULT_BOT_A,
    DEFAULT_BOT_B,
    "Random Bot",
    "Heuristic Bot",
)
BOT_LABELS = {
    DEFAULT_BOT_A: "Đăng poker Bot - NFSP / Deep RL - PyTorch",
    DEFAULT_BOT_B: "Poker-Bot-RL Bot - Tabular Q-Learning - Q-table",
    "Random Bot": "Random Bot - Random Policy - Test only",
    "Heuristic Bot": "Heuristic Bot - Rule-based - Test only",
}


def _inject_arena_styles():
    st.markdown(
        """
        <style>
        .arena-subtitle {color: #aeb8c7; margin: -0.5rem 0 1rem;}
        .arena-vs {text-align: center; font-size: 2rem; font-weight: 800;
                   padding-top: 3.2rem; color: #f6c453;}
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(90deg, #16835d, #1aa36f);
            border: 1px solid #35c28a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_session_defaults():
    defaults = {
        "arena_bot_a_kind": DEFAULT_BOT_A,
        "arena_bot_b_kind": DEFAULT_BOT_B,
        "arena_a_q_path": str(preferred_q_table_path()),
        "arena_b_q_path": str(preferred_q_table_path()),
        "arena_a_nfsp_path": str(preferred_nfsp_model_path()),
        "arena_b_nfsp_path": str(preferred_nfsp_model_path()),
        "arena_a_nfsp_dir": str(DEFAULT_NFSP_DIR),
        "arena_b_nfsp_dir": str(DEFAULT_NFSP_DIR),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state["arena_bot_a_kind"] not in BOT_OPTIONS:
        st.session_state["arena_bot_a_kind"] = DEFAULT_BOT_A
    if st.session_state["arena_bot_b_kind"] not in BOT_OPTIONS:
        st.session_state["arena_bot_b_kind"] = DEFAULT_BOT_B

    previous_result = st.session_state.get("bot_arena_result")
    if previous_result and (
        previous_result.get("bot_a_name") not in BOT_OPTIONS
        or previous_result.get("bot_b_name") not in BOT_OPTIONS
    ):
        st.session_state.pop("bot_arena_result", None)


def _paths_from_state(role):
    return (
        st.session_state[f"arena_{role}_q_path"],
        st.session_state[f"arena_{role}_nfsp_path"],
        st.session_state[f"arena_{role}_nfsp_dir"],
    )


def _build_bot(kind, q_path, nfsp_path, nfsp_code_dir, strict=False):
    if kind == DEFAULT_BOT_A:
        return DangPokerBotPolicy(nfsp_path or None, nfsp_code_dir, strict=strict)
    if kind == DEFAULT_BOT_B:
        return PokerBotRLPolicy(q_path, strict=strict)
    if kind == "Heuristic Bot":
        return HeuristicBotPolicy()
    if kind == "Random Bot":
        return RandomBotPolicy()
    raise ValueError(f"Bot không được hỗ trợ: {kind}")


def _status_badge(status):
    if status == "Loaded":
        return "✅ Loaded"
    if status == "Ready":
        return "✅ Ready"
    if status == "Missing model":
        return "⚠️ Missing model"
    if status == "Incompatible Q-table":
        return "❌ Incompatible"
    if status == "Import error":
        return "❌ Import error"
    if status == "Load error":
        return "❌ Load error"
    return f"⚠️ {status}"


def _short_algorithm(metadata):
    if metadata["name"] == DEFAULT_BOT_A:
        return "NFSP / Deep RL"
    if metadata["name"] == DEFAULT_BOT_B:
        return "Tabular Q-Learning"
    return metadata["algorithm"]


def _short_model(metadata):
    if metadata["name"] == DEFAULT_BOT_A:
        return "PyTorch .pt"
    if metadata["name"] == DEFAULT_BOT_B:
        return "Q-table .npy"
    return metadata["model_type"]


def _render_hero_card(bot):
    metadata = bot.get_metadata()
    with st.container(border=True):
        st.markdown(f"### {metadata['name']}")
        st.write(_short_algorithm(metadata))
        st.caption(f"Model: {_short_model(metadata)}")
        st.markdown(f"**{_status_badge(metadata['status'])}**")


def _render_hero(bot_a, bot_b):
    left, middle, right = st.columns([5, 1, 5])
    with left:
        _render_hero_card(bot_a)
    with middle:
        st.markdown('<div class="arena-vs">VS</div>', unsafe_allow_html=True)
    with right:
        _render_hero_card(bot_b)


def _render_model_inputs(role, kind):
    role_name = "Bot A" if role == "a" else "Bot B"
    if kind == DEFAULT_BOT_A:
        st.text_input(
            f"Source folder {role_name}", key=f"arena_{role}_nfsp_dir"
        )
        st.text_input(f"Model path {role_name}", key=f"arena_{role}_nfsp_path")
    elif kind == DEFAULT_BOT_B:
        st.text_input(f"Model path {role_name}", key=f"arena_{role}_q_path")


def _comparison_row(role, metadata):
    return {
        "Role": role,
        "Bot": metadata["name"],
        "Algorithm": metadata["algorithm"],
        "Model type": metadata["model_type"],
        "Model path": metadata["model_path"] or "—",
        "Status": _status_badge(metadata["status"]),
    }


def _render_error_details(bot_a, bot_b):
    failed = [("Bot A", bot_a), ("Bot B", bot_b)]
    failed = [(role, bot) for role, bot in failed if not bot.is_ready]
    if not failed:
        return

    with st.expander("Xem chi tiết lỗi model"):
        for role, bot in failed:
            metadata = bot.get_metadata()
            st.markdown(
                f"**{role}: {metadata['name']} — {_status_badge(metadata['status'])}**"
            )
            if metadata["status"] == "Missing model":
                st.write("Không tìm thấy file:")
                st.code(metadata["model_path"] or "Chưa chọn model path")
            st.write(getattr(bot, "error_message", metadata["status"]))

            if isinstance(bot, DangPokerBotPolicy):
                st.write(
                    "Hiện chưa tìm thấy PyTorch checkpoint `.pt` cho Đăng poker Bot."
                )
                st.write(
                    "Đăng poker Bot chưa thể đấu cho đến khi có model hợp lệ."
                )
                st.code(
                    'cd "C:\\REL301m\\code\\poker\\Đăng poker"\npython train.py',
                    language="powershell",
                )
                st.caption(
                    "Nếu train.py đang lỗi NFSP, project Đăng poker cần được sửa/train "
                    "trước; Bot Arena không sửa sâu folder này."
                )
            elif isinstance(bot, PokerBotRLPolicy):
                st.write(
                    "Q-table phải khớp state_size/action_size của environment hiện tại."
                )
                st.code(
                    'cd "C:\\REL301m\\code\\poker\\Poker-Bot-RL\\poker-bot"\n'
                    "python train.py --episodes 10000 --eval-interval 1000 --seed 42 "
                    "--output-dir artifacts/current",
                    language="powershell",
                )


def _result_comparison(result):
    counts_a = result["action_counts"]["bot_a"]
    counts_b = result["action_counts"]["bot_b"]
    return [
        {
            "Metric": "Wins",
            result["bot_a_name"]: result["bot_a_wins"],
            result["bot_b_name"]: result["bot_b_wins"],
        },
        {
            "Metric": "Win rate",
            result["bot_a_name"]: f"{result['bot_a_win_rate']:.1%}",
            result["bot_b_name"]: f"{result['bot_b_win_rate']:.1%}",
        },
        {
            "Metric": "Avg reward",
            result["bot_a_name"]: f"{result['avg_reward_a']:.3f}",
            result["bot_b_name"]: f"{result['avg_reward_b']:.3f}",
        },
        {
            "Metric": "Bet/Raise count",
            result["bot_a_name"]: counts_a["1"],
            result["bot_b_name"]: counts_b["1"],
        },
        {
            "Metric": "Call/Check count",
            result["bot_a_name"]: counts_a["0"],
            result["bot_b_name"]: counts_b["0"],
        },
        {
            "Metric": "Fold count",
            result["bot_a_name"]: counts_a["2"],
            result["bot_b_name"]: counts_b["2"],
        },
    ]


def render_bot_arena_tab():
    _inject_arena_styles()
    _set_session_defaults()

    kind_a = st.session_state["arena_bot_a_kind"]
    kind_b = st.session_state["arena_bot_b_kind"]
    bot_a = _build_bot(kind_a, *_paths_from_state("a"))
    bot_b = _build_bot(kind_b, *_paths_from_state("b"))

    st.title("🤖 Bot Arena - Đăng Bot vs Poker-Bot-RL Bot")
    st.markdown(
        '<div class="arena-subtitle">Hai bot từ hai folder khác nhau thi đấu '
        "trên cùng ShortDeckPokerEnv của branch main để đảm bảo cùng luật chơi.</div>",
        unsafe_allow_html=True,
    )
    _render_hero(bot_a, bot_b)

    with st.expander("⚙️ Match Setup"):
        left, right = st.columns(2)
        with left:
            kind_a = st.selectbox(
                "Select Bot A",
                BOT_OPTIONS,
                format_func=BOT_LABELS.get,
                key="arena_bot_a_kind",
            )
            _render_model_inputs("a", kind_a)
        with right:
            kind_b = st.selectbox(
                "Select Bot B",
                BOT_OPTIONS,
                format_func=BOT_LABELS.get,
                key="arena_bot_b_kind",
            )
            _render_model_inputs("b", kind_b)

        c1, c2, c3 = st.columns(3)
        num_games = c1.number_input(
            "Number of games", min_value=1, max_value=10000, value=500
        )
        seed = c2.number_input("Seed", value=42, step=1)
        alternate = c3.checkbox("Luân phiên vị trí", value=True)

    bot_a = _build_bot(kind_a, *_paths_from_state("a"))
    bot_b = _build_bot(kind_b, *_paths_from_state("b"))
    metadata_a = bot_a.get_metadata()
    metadata_b = bot_b.get_metadata()

    st.subheader("Match Setup")
    st.dataframe(
        [
            _comparison_row("Bot A", metadata_a),
            _comparison_row("Bot B", metadata_b),
        ],
        use_container_width=True,
        hide_index=True,
    )
    _render_error_details(bot_a, bot_b)

    can_start = bot_a.is_ready and bot_b.is_ready
    button_label = f"Start: {metadata_a['name']} vs {metadata_b['name']}"
    if not can_start:
        st.warning(
            "Chưa thể bắt đầu: cả hai bot phải có trạng thái Loaded/Ready. "
            "Mở phần chi tiết lỗi model để xem cách xử lý."
        )

    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=not can_start,
    ):
        st.session_state.pop("bot_arena_result", None)
        progress = st.progress(0, text="Đang chuẩn bị trận đấu...")

        def update_progress(done, total):
            progress.progress(done / total, text=f"Đã chạy {done}/{total} game")

        try:
            with st.spinner("Hai bot đang thi đấu trên environment của main..."):
                result = run_bot_match(
                    bot_a,
                    bot_b,
                    num_games=int(num_games),
                    seed=int(seed),
                    alternate_positions=alternate,
                    progress_callback=update_progress,
                )
        except Exception as exc:
            progress.empty()
            st.error(
                f"Trận đấu đã dừng, không dùng bot fallback. Chi tiết: {exc}"
            )
            return
        st.session_state.bot_arena_result = result
        progress.empty()

    result = st.session_state.get("bot_arena_result")
    if not result:
        return

    st.subheader("Match Result")
    metrics = st.columns(7)
    metrics[0].metric("Bot A win rate", f"{result['bot_a_win_rate']:.1%}")
    metrics[1].metric("Bot B win rate", f"{result['bot_b_win_rate']:.1%}")
    metrics[2].metric("Draw rate", f"{result['draw_rate']:.1%}")
    metrics[3].metric("Avg reward A", f"{result['avg_reward_a']:.3f}")
    metrics[4].metric("Avg reward B", f"{result['avg_reward_b']:.3f}")
    metrics[5].metric("Avg pot", f"{result['avg_pot']:.2f}")
    metrics[6].metric("Timeout", result["timeouts"])

    st.info(
        f"Bot A used: {result['bot_a_name']} - "
        f"{result['bot_a_metadata']['algorithm']}\n\n"
        f"Bot B used: {result['bot_b_name']} - "
        f"{result['bot_b_metadata']['algorithm']}"
    )
    st.dataframe(
        _result_comparison(result), use_container_width=True, hide_index=True
    )

    chart_result, chart_a, chart_b = st.columns(3)
    with chart_result:
        st.markdown("#### Win / Draw / Loss")
        st.bar_chart(
            {
                "Games": [
                    result["bot_a_wins"],
                    result["draws"],
                    result["bot_b_wins"],
                ]
            }
        )
    with chart_a:
        st.markdown("#### Actions Bot A")
        st.bar_chart(
            {
                "Actions": [
                    result["action_counts"]["bot_a"][str(i)] for i in range(3)
                ]
            }
        )
    with chart_b:
        st.markdown("#### Actions Bot B")
        st.bar_chart(
            {
                "Actions": [
                    result["action_counts"]["bot_b"][str(i)] for i in range(3)
                ]
            }
        )
    st.caption("Action 0/1/2: Check-Call / Bet-Raise / Fold.")

    st.subheader("Replay gần nhất")
    for replay in reversed(result["replays"]):
        with st.expander(
            f"Game #{replay['game']} — {replay['winner_bot']} — pot={replay['pot']}"
        ):
            st.write("**Bot A:**", replay["bot_a"])
            st.write("**Bot B:**", replay["bot_b"])
            st.write("**Winner:**", replay["winner_bot"])
            st.write("**Pot:**", replay["pot"])
            st.write("**Community cards:**", replay["community_cards"])
            st.dataframe(replay["actions"], use_container_width=True, hide_index=True)

    payload = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "Download JSON kết quả",
        data=payload,
        file_name="bot_arena_result.json",
        mime="application/json",
    )
