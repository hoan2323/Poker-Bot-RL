"""Versioned metadata for model files produced by train.py."""

import json
from datetime import datetime, timezone
from pathlib import Path


TRAINING_FORMAT_VERSION = 2
METADATA_FILENAME = "training_metadata.json"


def metadata_path(base_dir):
    return Path(base_dir) / METADATA_FILENAME


def write_training_metadata(base_dir, episodes, q_table_states):
    metadata = {
        "format_version": TRAINING_FORMAT_VERSION,
        "environment": "ShortDeckPokerEnv",
        "terminal_result_adapter": "game_results.resolve_terminal_result",
        "episodes": int(episodes),
        "q_table_states": int(q_table_states),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
    return metadata
