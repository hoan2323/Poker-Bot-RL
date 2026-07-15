"""Create concise training and evaluation reports from saved artifacts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import EPSILON, EPSILON_DECAY, MIN_EPSILON
from training_artifacts import require_current_training_metadata


BASE_DIR = Path(__file__).resolve().parent
TRAINING_CURVE_PATH = BASE_DIR / "training_curve.png"
EVALUATION_REPORT_PATH = BASE_DIR / "evaluation_summary.json"
EVALUATION_FIGURE_PATH = BASE_DIR / "evaluation_report.png"


def load_array(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing training artifact: {path.name}")
    return np.asarray(np.load(path), dtype=np.float64)


def rolling_mean(values, window):
    if not len(values):
        return values
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def create_training_curve():
    rewards = load_array("rewards.npy")
    win_rates = load_array("win_rates.npy")
    draw_rates = load_array("draw_rates.npy")
    loss_rates = load_array("loss_rates.npy")
    window = min(10_000, len(rewards))

    smoothed_reward = rolling_mean(rewards, window)
    smoothed_win_rate = rolling_mean(win_rates, window)
    smoothed_draw_rate = rolling_mean(draw_rates, window)
    smoothed_loss_rate = rolling_mean(loss_rates, window)
    episodes = np.arange(window, len(rewards) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(episodes, smoothed_win_rate, label="Win", color="#16803c")
    axes[0].plot(episodes, smoothed_draw_rate, label="Draw", color="#b98b00")
    axes[0].plot(episodes, smoothed_loss_rate, label="Loss", color="#ae2f2f")
    axes[0].set_title(f"Outcome rates (rolling {window:,} episodes)")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(episodes, smoothed_reward, color="#315f9d")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("Average reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(TRAINING_CURVE_PATH, dpi=150)
    plt.close(fig)

    final_epsilon = max(MIN_EPSILON, EPSILON * (EPSILON_DECAY ** len(rewards)))
    print("Training highlights")
    print(f"  Episodes: {len(rewards):,}")
    print(f"  Final epsilon: {final_epsilon:.4f}")
    print(f"  Last smoothed win rate: {smoothed_win_rate[-1]:.2%}")
    print(f"  Last smoothed reward: {smoothed_reward[-1]:.4f}")
    print(f"  Saved: {TRAINING_CURVE_PATH.name}")


def load_evaluation_report(path=EVALUATION_REPORT_PATH):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_highlights(matchups):
    q_matchups = [row for row in matchups if row["label"].startswith("q_vs_")]
    if not q_matchups:
        return []

    best = max(q_matchups, key=lambda row: row["win_rate"])
    weakest = min(q_matchups, key=lambda row: row["win_rate"])
    highlights = [
        f"Best Q matchup: {best['label']} ({best['win_rate']:.2%} win rate).",
        f"Weakest Q matchup: {weakest['label']} ({weakest['win_rate']:.2%} win rate).",
    ]

    for row in q_matchups:
        wins = row["wins"]
        fold_dependency = row["fold_wins"] / wins if wins else 0.0
        showdowns = row["showdown_ends"]
        showdown_record = (
            f"{row['showdown_wins']}W/{row['showdown_losses']}L/{row['showdown_draws']}D"
        )
        highlights.append(
            f"{row['label']}: showdown {showdown_record} across {showdowns} games; "
            f"{fold_dependency:.1%} of wins came from folds."
        )
    return highlights


def create_evaluation_report(report, output_path=EVALUATION_FIGURE_PATH):
    matchups = report.get("matchups", [])
    if not matchups:
        raise ValueError("evaluation_summary.json contains no matchup results.")

    labels = [row["label"].replace("_", " ") for row in matchups]
    win_rates = [row["win_rate"] for row in matchups]
    rewards = [row["avg_reward"] for row in matchups]
    showdown_rates = [row["showdown_only_win_rate"] for row in matchups]
    positions = np.arange(len(matchups))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(positions, win_rates, color="#16803c", label="Overall win rate")
    axes[0].barh(positions, showdown_rates, color="#315f9d", alpha=0.75, label="Showdown win rate")
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Rate")
    axes[0].set_title("Overall vs showdown performance")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.2)

    colors = ["#16803c" if reward >= 0 else "#ae2f2f" for reward in rewards]
    axes[1].barh(positions, rewards, color=colors)
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set_yticks(positions, labels)
    axes[1].set_xlabel("Average reward")
    axes[1].set_title("Average reward by matchup")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Evaluation highlights")
    for highlight in evaluation_highlights(matchups):
        print(f"  {highlight}")
    print(f"  Saved: {output_path.name}")


def main():
    require_current_training_metadata(BASE_DIR)
    create_training_curve()
    report = load_evaluation_report()
    if report is None:
        print("Evaluation report not found. Run evaluate.py after training to create it.")
        return
    create_evaluation_report(report)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(
            f"Cannot create plots: {exc}\n"
            "Run train.py first, then evaluate.py, to generate compatible artifacts."
        ) from None
