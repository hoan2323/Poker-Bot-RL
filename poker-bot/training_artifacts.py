"""Versioned metadata for model files produced by train.py."""

import json
from datetime import datetime, timezone
from pathlib import Path


TRAINING_FORMAT_VERSION = 3
METADATA_FILENAME = "training_metadata.json"


def metadata_path(base_dir):
    return Path(base_dir) / METADATA_FILENAME


def write_training_metadata(
    base_dir,
    episodes,
    q_table_states,
    state_size=186,
    action_size=3,
    state_key_size=15,
):
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "format_version": TRAINING_FORMAT_VERSION,
        "environment": "ShortDeckPokerEnv",
        "env_class": "ShortDeckPokerEnv",
        "terminal_result_adapter": "game_results.resolve_terminal_result",
        "episodes": int(episodes),
        "q_table_states": int(q_table_states),
        "state_size": int(state_size),
        "state_key_size": int(state_key_size),
        "action_size": int(action_size),
        "q_table_shape": [int(q_table_states), int(action_size)],
        "created_at": created_at,
        "created_at_utc": created_at,
        "note": "Sparse dict Q-table; q_table_shape is its logical shape.",
    }
    metadata_path(base_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def require_current_training_metadata(base_dir):
    path = metadata_path(base_dir)
    if not path.exists():
        raise RuntimeError("Q-table needs retraining with the current terminal reward adapter.")

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("training_metadata.json is unreadable. Retrain the Q-table.") from exc

    if metadata.get("format_version") != TRAINING_FORMAT_VERSION:
        raise RuntimeError("Q-table was trained with an incompatible format. Retrain it.")
    if metadata.get("state_size") != 186 or metadata.get("action_size") != 3:
        raise RuntimeError("Q-table metadata has an incompatible state/action size. Retrain it.")
    return metadata
