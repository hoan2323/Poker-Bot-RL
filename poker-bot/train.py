import argparse
import random
from pathlib import Path
import numpy as np

from environment import ShortDeckPokerEnv
from game_results import resolve_terminal_result
from q_learning_agent import QLearningAgent
from opponent_model import OpponentModel
from training_artifacts import write_training_metadata
from evaluate import (
    always_bet_policy,
    call_station_policy,
    heuristic_policy,
    observation_for_player,
    random_policy,
)


EPISODES = 500000
ALPHA = 0.15
GAMMA = 0.95
EPSILON = 0.45
EPSILON_DECAY = 0.99999
MIN_EPSILON = 0.02
LOG_INTERVAL = 10000


OPPONENT_POLICIES = {
    "heuristic": heuristic_policy,
    "call_station": call_station_policy,
    "random": random_policy,
    "always_bet": always_bet_policy,
}

OPPONENT_WEIGHTS = {
    "heuristic": 0.50,
    "call_station": 0.35,
    "always_bet": 0.10,
    "random": 0.05,
}


def choose_training_opponent(episode):
    """Sample opponents by configured weights to reduce overfitting to weak/random play."""
    names = list(OPPONENT_POLICIES.keys())
    weights = [OPPONENT_WEIGHTS[name] for name in names]
    name = random.choices(names, weights=weights, k=1)[0]
    return name, OPPONENT_POLICIES[name]


USE_REWARD_SHAPING = True


def shaping_reward(agent, state, action, valid_actions):
    """
    Tiny Player 0-only shaping.
    Keeps terminal env reward honest (+pot / -pot / 0).

    Purpose:
    - push strong hands toward value bets in short deck
    - encourage medium hands to bet/semi-bluff more often
    - discourage weak calls without over-penalizing weak bluffs

    Important:
    - shaping stays small so terminal pot reward still dominates
    """
    if not USE_REWARD_SHAPING:
        return 0.0

    state_key = agent.get_state_key(state)
    made_hand_rank = state_key[0]
    best_current_hand_rank = state_key[1]
    two_pair_flag = state_key[3]
    trips_flag = state_key[4]
    flush_draw = state_key[5]
    straight_draw = state_key[6]
    top_pair = state_key[7]
    overpair = state_key[8]
    pot_bucket = state_key[10]
    has_active_bet = state_key[11]
    round_index = state_key[13]

    reward = 0.0

    top_pair_only = bool(top_pair and made_hand_rank == 1 and not two_pair_flag and not trips_flag and not overpair)
    strong_hand = (
        best_current_hand_rank >= 2
        or two_pair_flag
        or trips_flag
        or overpair
    )
    medium_hand = (
        not strong_hand
        and (
            best_current_hand_rank >= 1
            or top_pair_only
            or made_hand_rank >= 2
            or flush_draw
            or straight_draw
        )
    )
    weak_hand = not strong_hand and not medium_hand

    if has_active_bet == 0 and 1 in valid_actions:
        if weak_hand:
            if action == 1:
                reward -= 0.080 if round_index >= 1 else 0.050
            elif action == 0:
                reward += 0.010

        elif medium_hand:
            if action == 1:
                reward += 0.080 if round_index >= 1 else 0.040
            elif action == 0:
                reward -= 0.030 if round_index >= 1 else 0.010

        elif strong_hand:
            if action == 1:
                reward += 0.500
            elif action == 0:
                reward -= 0.300

        if round_index >= 2 and strong_hand:
            if action == 1:
                reward += 0.500
            elif action == 0:
                reward -= 0.400

    # Facing bet
    if has_active_bet == 1 and 2 in valid_actions:
        if weak_hand:
            if action == 0:
                reward -= 0.040 if pot_bucket <= 1 else 0.120
            elif action == 2 and pot_bucket >= 2:
                reward += 0.020

        elif medium_hand:
            if action == 0 and pot_bucket <= 1:
                reward += 0.005
            elif action == 0 and pot_bucket >= 2:
                reward -= 0.080
            elif action == 2 and pot_bucket <= 1:
                reward -= 0.008

        elif strong_hand:
            if action == 0:
                reward += 0.100 if pot_bucket <= 1 else 0.150
            elif action == 2:
                reward -= 0.250 if pot_bucket <= 1 else 0.350

    return reward


def run_training(episodes=EPISODES, log_interval=LOG_INTERVAL, output_dir=None, seed=42):
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero")

    random.seed(seed)
    np.random.seed(seed)
    env = ShortDeckPokerEnv()
    agent = QLearningAgent(
        action_size=3,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        random_tie_break=True,
    )

    rewards = []
    win_rates = []
    draw_rates = []
    loss_rates = []

    wins = 0
    draws = 0
    losses = 0

    opponent_counts = {name: 0 for name in OPPONENT_POLICIES}
    opponent_models = {name: OpponentModel() for name in OPPONENT_POLICIES}

    for episode in range(episodes):
        opponent_name, opponent_policy = choose_training_opponent(episode)
        opponent_counts[opponent_name] += 1

        starting_player = episode % 2
        state = env.reset(starting_player=starting_player)
        opponent_model = opponent_models[opponent_name]
        done = False
        total_reward = 0
        pending_transition = None

        while not done:
            acting_player = env.current_player
            valid_actions = env.get_valid_actions()
            policy_state = observation_for_player(env, acting_player)

            opponent_profile = opponent_model.profile_bucket()

            if acting_player == 0:
                if pending_transition is not None:
                    pending_transition["next_state"] = policy_state
                    pending_transition["next_opponent_profile"] = opponent_profile
                    pending_transition["valid_next_actions"] = valid_actions
                    agent.update(
                        pending_transition["state"],
                        pending_transition["action"],
                        pending_transition["reward"],
                        pending_transition["next_state"],
                        False,
                        pending_transition["valid_next_actions"],
                        opponent_profile=pending_transition["opponent_profile"],
                        next_opponent_profile=pending_transition["next_opponent_profile"],
                    )
                    pending_transition = None

                action = agent.choose_action(
                    policy_state,
                    valid_actions,
                    opponent_profile=opponent_profile,
                )
            else:
                action = opponent_policy(policy_state, valid_actions)

            next_state, step_reward, done, info = env.step(action)
            reward, _, _ = resolve_terminal_result(env, step_reward)

            if acting_player == 1:
                opponent_model.record_action(valid_actions, action)

            if acting_player == 0:
                training_reward = reward + shaping_reward(agent, policy_state, action, valid_actions)
                pending_transition = {
                    "state": policy_state,
                    "action": action,
                    "reward": training_reward,
                    "opponent_profile": opponent_profile,
                    "next_state": None,
                    "next_opponent_profile": opponent_model.profile_bucket(),
                    "valid_next_actions": [],
                }
            elif pending_transition is not None:
                pending_transition["reward"] += reward

            if done and pending_transition is not None:
                next_policy_state = observation_for_player(env, 0)
                agent.update(
                    pending_transition["state"],
                    pending_transition["action"],
                    pending_transition["reward"],
                    next_policy_state,
                    True,
                    [],
                    opponent_profile=pending_transition["opponent_profile"],
                    next_opponent_profile=opponent_model.profile_bucket(),
                )
                pending_transition = None

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

        games_played = episode + 1
        win_rates.append(wins / games_played)
        draw_rates.append(draws / games_played)
        loss_rates.append(losses / games_played)

        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

        if games_played % log_interval == 0:
            recent_rewards = rewards[-log_interval:]
            avg_reward = float(np.mean(recent_rewards))
            recent_wins = sum(1 for value in recent_rewards if value > 0)
            recent_draws = sum(1 for value in recent_rewards if value == 0)
            recent_win_rate = recent_wins / len(recent_rewards)
            recent_draw_rate = recent_draws / len(recent_rewards)

            print(
                f"Episode {games_played}/{episodes} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Win Rate: {recent_win_rate:.4f} | "
                f"Draw Rate: {recent_draw_rate:.4f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Q-States: {len(agent.q_table)} | "
                f"Opponents: {opponent_counts}"
            )

    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    agent.save(output_dir / "q_table.npy")
    np.save(output_dir / "rewards.npy", np.array(rewards, dtype=np.float64))
    np.save(output_dir / "win_rates.npy", np.array(win_rates, dtype=np.float64))
    np.save(output_dir / "draw_rates.npy", np.array(draw_rates, dtype=np.float64))
    np.save(output_dir / "loss_rates.npy", np.array(loss_rates, dtype=np.float64))
    write_training_metadata(output_dir, episodes, len(agent.q_table))
    print(f"Compatible Q-table saved to {output_dir / 'q_table.npy'}")

    return agent, rewards, win_rates, draw_rates, loss_rates


def parse_args():
    parser = argparse.ArgumentParser(description="Train the tabular Q-learning poker bot.")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-interval",
        "--log-interval",
        dest="log_interval",
        type=int,
        default=LOG_INTERVAL,
        help="Print rolling training statistics every N episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for q_table.npy, metrics, and metadata.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.episodes, args.log_interval, args.output_dir, args.seed)
