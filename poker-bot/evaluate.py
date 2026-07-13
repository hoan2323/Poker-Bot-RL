import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from environment import ShortDeckPokerEnv
from game_results import normalized_step_info, resolve_terminal_result
from q_learning_agent import QLearningAgent
from opponent_model import OpponentModel
from training_artifacts import require_current_training_metadata


EVAL_GAMES = 5000
RANKS = [10, 11, 12, 13, 14]
BASE_DIR = Path(__file__).resolve().parent
Q_TABLE_PATH = BASE_DIR / "q_table.npy"
EVALUATION_REPORT_PATH = BASE_DIR / "evaluation_summary.json"


def decode_card(card_id: int):
    if card_id == -1:
        return None
    if card_id < 0 or card_id >= len(RANKS) * 4:
        return 14, 0
    rank = RANKS[card_id // 4]
    suit = card_id % 4
    return rank, suit


def observation_for_player(env, player):
    return env.get_state(player)


def visible_board_cards(state):
    return np.flatnonzero(np.asarray(state)[80:100] > 0.5).astype(int).tolist()


def hole_cards(state):
    return np.flatnonzero(np.asarray(state)[:20] > 0.5).astype(int).tolist()


def visible_ranks(state):
    return [decode_card(card)[0] for card in visible_board_cards(state)]


def visible_suits(state):
    return [decode_card(card)[1] for card in visible_board_cards(state)]


def has_pair(cards: List[int]) -> bool:
    ranks = [decode_card(card)[0] for card in cards if card != -1]
    return len(ranks) != len(set(ranks))


def high_card_bucket(cards: List[int]) -> int:
    ranks = [decode_card(card)[0] for card in cards if card != -1]
    if not ranks:
        return 0
    high = max(ranks)
    if high <= 9:
        return 0
    if high in (10, 11):
        return 1
    if high in (12, 13):
        return 2
    return 3


def board_flush_danger(state) -> bool:
    suits = visible_suits(state)
    for suit in set(suits):
        if suits.count(suit) >= 3:
            return True
    return False


def board_straight_danger(state) -> bool:
    ranks = sorted(set(visible_ranks(state)))
    if len(ranks) < 3:
        return False
    for start_idx in range(len(ranks) - 2):
        for end_idx in range(start_idx + 2, len(ranks)):
            window = ranks[start_idx : end_idx + 1]
            if len(window) >= 3 and max(window) - min(window) <= 4:
                return True
    return False


def made_hand_strength(state) -> bool:
    cards = hole_cards(state) + visible_board_cards(state)
    return has_pair(cards) or board_flush_danger(state) or board_straight_danger(state)


def made_hand_or_strong_draw(state):
    """Check if current hand has a pair, top pair, flush draw, straight draw, or strong overcard."""
    hole = hole_cards(state)
    board = visible_board_cards(state)
    all_known = hole + board
    if has_pair(all_known):
        return True
    hole_ranks = [decode_card(card)[0] for card in hole]
    hole_high = max(hole_ranks)
    board_ranks = visible_ranks(state)
    if board_ranks:
        max_board = max(board_ranks)
        # overcard to board
        for hr in hole_ranks:
            if hr > max_board:
                return True
    # flush draw
    suits = [decode_card(card)[1] for card in all_known]
    for suit in set(suits):
        if suits.count(suit) >= 4:
            return True
    # straight draw
    all_ranks = set(hole_ranks + board_ranks)
    if len(all_ranks) >= 4:
        search = set(all_ranks)
        if 14 in search:
            search.add(1)
        sorted_r = sorted(search)
        for i in range(len(sorted_r) - 3):
            window = sorted_r[i:i + 4]
            if len(window) == 4 and window[3] - window[0] <= 4:
                return True
    return False


def very_strong_hand(state):
    """Check if hand has at least two pair, or top pair with strong kicker."""
    hole = hole_cards(state)
    board = visible_board_cards(state)
    all_known = hole + board
    ranks = [decode_card(card)[0] for card in all_known]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # two pair or better
    pairs = sum(1 for c in counts.values() if c >= 2)
    if pairs >= 2:
        return True
    # top pair with at least J kicker
    board_ranks = visible_ranks(state)
    if board_ranks:
        max_board = max(board_ranks)
        if counts.get(max_board, 0) >= 2:
            hole_ranks = [decode_card(card)[0] for card in hole]
            for hr in hole_ranks:
                if hr == max_board and hr >= 11:
                    return True
    return False


def heuristic_policy_for_state(state, valid_actions):
    hole = hole_cards(state)
    board_cards = visible_board_cards(state)
    all_known = hole + board_cards
    hole_ranks = [decode_card(card)[0] for card in hole]
    hole_high = max(hole_ranks)

    strong = very_strong_hand(state)
    decent = made_hand_or_strong_draw(state)

    if 0 in valid_actions and 2 in valid_actions:
        # Facing bet
        if strong:
            return 0  # call
        if decent:
            return 0  # call
        if hole_high >= 12:  # K or A high
            return 0
        return 2  # fold weak

    # No active bet — can check or bet
    if strong:
        return 1 if 1 in valid_actions else 0  # bet
    if decent:
        # bet with decent hand sometimes, but randomize a bit
        if hole_high >= 13 or (board_cards and len(board_cards) >= 4):
            return 1 if 1 in valid_actions else 0
        return 0
    # weak hand: mostly check, occasionally bluff
    if random.random() < 0.15:
        return 1 if 1 in valid_actions else 0
    return 0


def random_policy(state, valid_actions):
    return random.choice(valid_actions)


def always_bet_policy(state, valid_actions):
    return 1 if 1 in valid_actions else valid_actions[0]


def always_call_policy(state, valid_actions):
    return 0 if 0 in valid_actions else valid_actions[0]


def call_station_policy(state, valid_actions):
    return 0 if 0 in valid_actions else valid_actions[0]


def heuristic_policy(state, valid_actions):
    return heuristic_policy_for_state(state, valid_actions)


def mixed_opponent_policy(state, valid_actions):
    return random.choice([random_policy, call_station_policy, heuristic_policy])(state, valid_actions)


def hand_strength_bucket(state):
    hole = hole_cards(state)
    board = visible_board_cards(state)
    all_known = hole + board
    ranks = [decode_card(card)[0] for card in all_known]
    counts = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1

    pair_count = sum(1 for count in counts.values() if count >= 2)
    if any(count >= 3 for count in counts.values()):
        return "strong"
    if pair_count >= 2:
        return "strong"
    if made_hand_or_strong_draw(state):
        return "medium"
    return "weak"


def init_action_diagnostics():
    return {
        "weak": {"actions": 0, "bet": 0, "call": 0, "fold": 0, "reward_sum": 0.0, "large_pot_loss": 0},
        "medium": {"actions": 0, "bet": 0, "call": 0, "fold": 0, "reward_sum": 0.0, "large_pot_loss": 0},
        "strong": {"actions": 0, "bet": 0, "call": 0, "fold": 0, "reward_sum": 0.0, "large_pot_loss": 0},
    }


def q_policy(agent):
    def policy(state, valid_actions, opponent_profile=0):
        return agent.choose_action(
            state,
            valid_actions,
            opponent_profile=opponent_profile,
        )

    policy.uses_opponent_model = True
    return policy


@dataclass
class EpisodeResult:
    winner: int | None
    end_reason: str
    reward: float
    pot: int
    starting_player: int
    fold_ending: bool
    showdown_ending: bool
    folds_when_facing_bet: int
    faced_bet_count: int
    large_pot_game: bool
    large_pot_win: bool
    large_pot_loss: bool
    action_diagnostics: dict = field(default_factory=dict)


def run_episode(env, policy0, policy1, starting_player=0, opponent_model=None):
    state = env.reset(starting_player=starting_player)
    done = False
    total_reward = 0
    final_info = env._get_info()
    folds_when_facing_bet = 0
    faced_bet_count = 0
    action_diagnostics = init_action_diagnostics()
    agent_action_strengths = []
    if opponent_model is None:
        opponent_model = OpponentModel()

    while not done:
        acting_player = env.current_player
        valid_actions = env.get_valid_actions()
        policy_state = observation_for_player(env, acting_player)
        policy = policy0 if acting_player == 0 else policy1
        if acting_player == 0 and getattr(policy, "uses_opponent_model", False):
            action = policy(
                policy_state,
                valid_actions,
                opponent_profile=opponent_model.profile_bucket(),
            )
        else:
            action = policy(policy_state, valid_actions)

        if acting_player == 0:
            strength = hand_strength_bucket(policy_state)
            agent_action_strengths.append(strength)
            action_diagnostics[strength]["actions"] += 1
            if action == 1:
                action_diagnostics[strength]["bet"] += 1
            elif action == 2:
                action_diagnostics[strength]["fold"] += 1
            elif 0 in valid_actions and 2 in valid_actions:
                action_diagnostics[strength]["call"] += 1

        if acting_player == 0 and 0 in valid_actions and 2 in valid_actions:
            faced_bet_count += 1
            if action == 2:
                folds_when_facing_bet += 1

        next_state, step_reward, done, info = env.step(action)
        reward, info = normalized_step_info(env, info, step_reward)
        if acting_player == 1:
            opponent_model.record_action(valid_actions, action)
        state = next_state
        total_reward += reward
        final_info = info

    final_pot = final_info["pot"]
    terminal_reward, winner, end_reason = resolve_terminal_result(env)
    total_reward = terminal_reward if done else total_reward

    if agent_action_strengths:
        reward_share = total_reward / len(agent_action_strengths)
        for strength in agent_action_strengths:
            action_diagnostics[strength]["reward_sum"] += reward_share
        if final_pot >= 5 and total_reward < 0:
            action_diagnostics[agent_action_strengths[-1]]["large_pot_loss"] += 1

    return EpisodeResult(
        winner=winner,
        end_reason=end_reason,
        reward=total_reward,
        pot=final_pot,
        starting_player=starting_player,
        fold_ending=end_reason == "fold",
        showdown_ending=end_reason == "showdown",
        folds_when_facing_bet=folds_when_facing_bet,
        faced_bet_count=faced_bet_count,
        large_pot_game=final_pot >= 5,
        large_pot_win=final_pot >= 5 and total_reward > 0,
        large_pot_loss=final_pot >= 5 and total_reward < 0,
        action_diagnostics=action_diagnostics,
    )


def summarize_results(results):
    wins = sum(1 for r in results if r.reward > 0)
    losses = sum(1 for r in results if r.reward < 0)
    draws = sum(1 for r in results if r.reward == 0)
    fold_ends = sum(1 for r in results if r.fold_ending)
    showdown_ends = sum(1 for r in results if r.showdown_ending)
    fold_wins = sum(1 for r in results if r.fold_ending and r.reward > 0)
    fold_losses = sum(1 for r in results if r.fold_ending and r.reward < 0)
    showdown_wins = sum(1 for r in results if r.showdown_ending and r.reward > 0)
    showdown_losses = sum(1 for r in results if r.showdown_ending and r.reward < 0)
    showdown_draws = sum(1 for r in results if r.showdown_ending and r.reward == 0)

    avg_reward = float(np.mean([r.reward for r in results]))
    avg_pot = float(np.mean([r.pot for r in results]))
    win_pots = [r.pot for r in results if r.reward > 0]
    loss_pots = [r.pot for r in results if r.reward < 0]
    avg_win_pot = float(np.mean(win_pots)) if win_pots else 0.0
    avg_loss_pot = float(np.mean(loss_pots)) if loss_pots else 0.0

    showdown_only_games = showdown_ends
    showdown_only_win_rate = showdown_wins / showdown_only_games if showdown_only_games else 0.0
    decisive_showdowns = showdown_wins + showdown_losses
    showdown_decisive_win_rate = showdown_wins / decisive_showdowns if decisive_showdowns else 0.0
    non_fold_games = len(results) - fold_ends
    win_rate_excluding_fold = showdown_wins / non_fold_games if non_fold_games else 0.0
    total_faced_bet = sum(r.faced_bet_count for r in results)
    total_folds_when_facing_bet = sum(r.folds_when_facing_bet for r in results)
    fold_rate_when_facing_bet = total_folds_when_facing_bet / total_faced_bet if total_faced_bet else 0.0

    large_pot_games = sum(1 for r in results if r.large_pot_game)
    large_pot_wins = sum(1 for r in results if r.large_pot_win)
    large_pot_losses = sum(1 for r in results if r.large_pot_loss)
    large_pot_win_rate = large_pot_wins / large_pot_games if large_pot_games else 0.0

    action_diagnostics = init_action_diagnostics()
    for result in results:
        for strength, stats in result.action_diagnostics.items():
            for key, value in stats.items():
                action_diagnostics[strength][key] += value

    action_summary = {}
    for strength, stats in action_diagnostics.items():
        actions = stats["actions"]
        action_summary[strength] = {
            "bet_rate": stats["bet"] / actions if actions else 0.0,
            "call_rate": stats["call"] / actions if actions else 0.0,
            "fold_rate": stats["fold"] / actions if actions else 0.0,
            "avg_reward": stats["reward_sum"] / actions if actions else 0.0,
            "large_pot_loss_count": stats["large_pot_loss"],
            "actions": actions,
        }

    by_position = {
        0: {
            "games": sum(1 for r in results if r.starting_player == 0),
            "wins": sum(1 for r in results if r.starting_player == 0 and r.reward > 0),
        },
        1: {
            "games": sum(1 for r in results if r.starting_player == 1),
            "wins": sum(1 for r in results if r.starting_player == 1 and r.reward > 0),
        },
    }
    for pos in by_position:
        games = by_position[pos]["games"]
        by_position[pos]["win_rate"] = by_position[pos]["wins"] / games if games else 0.0

    return {
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(results) if results else 0.0,
        "avg_reward": avg_reward,
        "fold_wins": fold_wins,
        "fold_losses": fold_losses,
        "fold_ends": fold_ends,
        "showdown_ends": showdown_ends,
        "showdown_wins": showdown_wins,
        "showdown_losses": showdown_losses,
        "showdown_draws": showdown_draws,
        "avg_pot": avg_pot,
        "avg_win_pot": avg_win_pot,
        "avg_loss_pot": avg_loss_pot,
        "fold_rate_when_facing_bet": fold_rate_when_facing_bet,
        "showdown_only_win_rate": showdown_only_win_rate,
        "showdown_decisive_win_rate": showdown_decisive_win_rate,
        "win_rate_excluding_fold": win_rate_excluding_fold,
        "large_pot_games": large_pot_games,
        "large_pot_wins": large_pot_wins,
        "large_pot_losses": large_pot_losses,
        "large_pot_win_rate": large_pot_win_rate,
        "action_diagnostics": action_summary,
        "by_position": by_position,
    }


def evaluate_matchup(label, policy0, policy1, env, games=EVAL_GAMES):
    results = []
    opponent_model = OpponentModel()
    for game_idx in range(games):
        starting_player = game_idx % 2
        results.append(run_episode(env, policy0, policy1, starting_player=starting_player, opponent_model=opponent_model))
    summary = summarize_results(results)
    summary["label"] = label
    return summary


def print_action_diagnostics(rows):
    print("\nPot Diagnostics:")
    print(f"{'Mode':<24} {'AvgPot':>8} {'AvgWinPot':>10} {'AvgLossPot':>11} {'LargePotWR':>10}")
    for row in rows:
        print(
            f"{row['label']:<24} "
            f"{row['avg_pot']:>8.3f} "
            f"{row['avg_win_pot']:>10.3f} "
            f"{row['avg_loss_pot']:>11.3f} "
            f"{row['large_pot_win_rate']:>10.4f}"
        )

    print("\nAction Diagnostics:")
    print(f"{'Mode':<24} {'Strength':<8} {'Actions':>8} {'Bet':>8} {'Call':>8} {'Fold':>8} {'AvgReward':>10} {'LargePotLoss':>12}")
    for row in rows:
        for strength, stats in row["action_diagnostics"].items():
            print(
                f"{row['label']:<24} "
                f"{strength:<8} "
                f"{stats['actions']:>8} "
                f"{stats['bet_rate']:>8.4f} "
                f"{stats['call_rate']:>8.4f} "
                f"{stats['fold_rate']:>8.4f} "
                f"{stats['avg_reward']:>10.4f} "
                f"{stats['large_pot_loss_count']:>12}"
            )


def print_table(rows):
    print("Evaluation Summary - 20-card Short Deck")
    print()
    print(
        f"{'Mode':<24} {'Games':>7} {'WinRate':>9} {'AvgReward':>11} "
        f"{'FoldWins':>10} {'FoldLosses':>11} {'FoldVsBet':>10} {'LargePotWR':>10} {'Showdown W/L/D':>15}"
    )
    for row in rows:
        showdown_record = f"{row['showdown_wins']}/{row['showdown_losses']}/{row['showdown_draws']}"
        print(
            f"{row['label']:<24} "
            f"{row['games']:>7} "
            f"{row['win_rate']:>9.4f} "
            f"{row['avg_reward']:>11.4f} "
            f"{row['fold_wins']:>10} "
            f"{row['fold_losses']:>11} "
            f"{row['fold_rate_when_facing_bet']:>10.4f} "
            f"{row['large_pot_win_rate']:>10.4f} "
            f"{showdown_record:>15}"
        )


def save_evaluation_report(rows, path=EVALUATION_REPORT_PATH):
    report = {
        "format_version": 2,
        "environment": "ShortDeckPokerEnv",
        "matchups": rows,
    }
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_trained_agent(q_table_path=Q_TABLE_PATH):
    require_current_training_metadata(BASE_DIR)
    agent = QLearningAgent(action_size=3, epsilon=0.0)
    agent.load(q_table_path)
    agent.epsilon = 0.0
    return agent


def run_evaluation():
    env = ShortDeckPokerEnv()
    agent = load_trained_agent()

    policies = {
        "random": random_policy,
        "call_station": call_station_policy,
        "heuristic": heuristic_policy,
        "always_bet": always_bet_policy,
        "mixed": mixed_opponent_policy,
        "q": q_policy(agent),
    }

    baseline_rows = [
        evaluate_matchup("q_vs_random", policies["q"], policies["random"], env),
        evaluate_matchup("q_vs_call_station", policies["q"], policies["call_station"], env),
        evaluate_matchup("q_vs_heuristic", policies["q"], policies["heuristic"], env),
        evaluate_matchup("always_bet_vs_heuristic", policies["always_bet"], policies["heuristic"], env),
        evaluate_matchup("mixed_opponent_eval", policies["q"], policies["mixed"], env),
    ]

    print_table(baseline_rows)
    print_action_diagnostics(baseline_rows)
    save_evaluation_report(baseline_rows)

    print(f"\nQ-table State Count: {len(agent.q_table)}")

    q_random = baseline_rows[0]
    q_call_station = baseline_rows[1]
    q_heuristic = baseline_rows[2]
    mixed = baseline_rows[4]

    print("\nInterpretation:")
    if q_heuristic["win_rate"] < q_random["win_rate"] and q_heuristic["win_rate"] < mixed["win_rate"]:
        print(
            "Agent performs well against random and mixed opponents, "
            "but is still weak against heuristic opponents."
        )
    elif q_call_station["win_rate"] < 0.45 or q_heuristic["win_rate"] < 0.45:
        print(
            "Agent is improving against baseline opponents, "
            "but still needs stronger value betting in difficult matchups."
        )
    else:
        print("Agent performs consistently across random, call-station, heuristic, and mixed opponents.")


if __name__ == "__main__":
    run_evaluation()
