"""Helpers for presenting terminal results consistently to callers."""


def resolve_terminal_result(env, step_reward=0.0):
    """Return Player 0 reward, winning player, and terminal reason.

    ShortDeckPokerEnv uses player ids for ``winner`` and Player 0's perspective
    for rewards. This helper remains as a stable boundary for training and UI.
    """
    if not getattr(env, "done", False):
        return float(step_reward), None, None

    winner = getattr(env, "winner", None)
    end_reason = "showdown" if getattr(env, "round", None) == 4 else "fold"

    pot = float(getattr(env, "pot", 0))
    reward = pot if winner == 0 else -pot if winner == 1 else 0.0
    return reward, winner, end_reason


def normalized_step_info(env, info, step_reward=0.0):
    """Return a UI-friendly info dictionary with normalized terminal fields."""
    reward, winner, end_reason = resolve_terminal_result(env, step_reward)
    normalized_info = dict(info or {})
    if end_reason is not None:
        normalized_info["winner"] = winner
        normalized_info["end_reason"] = end_reason
    return reward, normalized_info
