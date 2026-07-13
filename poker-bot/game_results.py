"""Compatibility helpers for interpreting ShortDeckPokerEnv terminal states.

The environment remains the source of game state. These helpers normalize its
showdown convention for the training, evaluation, and UI layers.
"""


def resolve_terminal_result(env, step_reward=0.0):
    """Return Player 0 reward, winning player, and terminal reason.

    ShortDeckPokerEnv exposes a comparison result at showdown (1 for its first
    hand, -1 for its second hand) rather than a player id, and its step reward
    is zero on that path. Fold results already use player ids. This adapter
    keeps callers independent from those two representations.
    """
    if not getattr(env, "done", False):
        return float(step_reward), None, None

    if getattr(env, "round", None) == 4:
        comparison = getattr(env, "winner", None)
        if comparison == 1:
            winner = 0
        elif comparison == -1:
            winner = 1
        else:
            winner = None
        end_reason = "showdown"
    else:
        winner = getattr(env, "winner", None)
        end_reason = "fold"

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
