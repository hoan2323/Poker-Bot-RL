import random
import numpy as np

from texas_holdenv import TexasHoldemEnv
from q_learning_agent import QLearningAgent
from opponent_model import OpponentModel
from evaluate import (
    always_bet_policy,
    call_station_policy,
    heuristic_policy,
    observation_for_player,
    random_policy,
)


EPISODES = 300000
ALPHA = 0.10
GAMMA = 0.95
EPSILON = 0.35
EPSILON_DECAY = 0.999985
MIN_EPSILON = 0.05
LOG_INTERVAL = 10000


OPPONENT_POLICIES = {
    "heuristic": heuristic_policy,
    "call_station": call_station_policy,
    "random": random_policy,
    "always_bet": always_bet_policy,
}

OPPONENT_WEIGHTS = {
    "heuristic": 0.40,
    "call_station": 0.30,
    "random": 0.20,
    "always_bet": 0.10,
}


def choose_training_opponent():
    names = list(OPPONENT_WEIGHTS.keys())
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

    reward = 0.0

    strong_hand = (
        best_current_hand_rank >= 2
        or two_pair_flag
        or trips_flag
        or overpair
        or (top_pair and made_hand_rank >= 1)
    )
    medium_hand = (
        not strong_hand
        and (
            best_current_hand_rank >= 1
            or top_pair
            or made_hand_rank >= 2
            or flush_draw
            or straight_draw
        )
    )
    weak_hand = not strong_hand and not medium_hand

    # No active bet: use small nudges only.
    if has_active_bet == 0 and 1 in valid_actions:
        if weak_hand:
            if action == 1:  # bluff bet
                reward -= 0.002
            elif action == 0:  # check
                reward += 0.003
        elif medium_hand:
            if action == 1:  # mixed value/semi-bluff
                reward += 0.020
            elif action == 0:  # check less often
                reward -= 0.003
        elif strong_hand:
            if action == 1:  # value bet
                reward += 0.075 if pot_bucket <= 1 else 0.090
            elif action == 0:  # slowplay less often
                reward -= 0.030 if pot_bucket <= 1 else 0.040

    # Facing bet: strongest shaping should be on bad weak calls in bigger pots.
    if has_active_bet == 1 and 2 in valid_actions:
        if weak_hand:
            if action == 0:  # call
                reward -= 0.030 if pot_bucket <= 1 else 0.050
            elif action == 2:  # fold
                reward += 0.020 if pot_bucket <= 1 else 0.035
        elif medium_hand:
            if action == 0 and pot_bucket <= 1:  # call smaller pots more often
                reward += 0.012
            elif action == 0 and pot_bucket >= 2:  # don't station big pots too much
                reward -= 0.006
            elif action == 2 and pot_bucket <= 1:
                reward -= 0.008
        elif strong_hand:
            if action == 0:  # call
                reward += 0.020 if pot_bucket <= 1 else 0.035
            elif action == 2:  # fold
                reward -= 0.090 if pot_bucket <= 1 else 0.120

    return reward


def run_training():
    env = TexasHoldemEnv()
    agent = QLearningAgent(
        action_size=3,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
    )

    rewards = []
    win_rates = []
    draw_rates = []
    loss_rates = []

    wins = 0
    draws = 0
    losses = 0

    opponent_counts = {name: 0 for name in OPPONENT_POLICIES}

    for episode in range(EPISODES):
        opponent_name, opponent_policy = choose_training_opponent()
        opponent_counts[opponent_name] += 1

        starting_player = episode % 2
        state, info = env.reset(options={"starting_player": starting_player})
        opponent_model = OpponentModel()
        terminated = False
        truncated = False
        total_reward = 0
        pending_transition = None

        while not terminated and not truncated:
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

            next_state, reward, terminated, truncated, info = env.step(action)

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

            if (terminated or truncated) and pending_transition is not None:
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

        if games_played % LOG_INTERVAL == 0:
            recent_rewards = rewards[-LOG_INTERVAL:]
            avg_reward = float(np.mean(recent_rewards))
            recent_wins = sum(1 for value in recent_rewards if value > 0)
            recent_draws = sum(1 for value in recent_rewards if value == 0)
            recent_win_rate = recent_wins / LOG_INTERVAL
            recent_draw_rate = recent_draws / LOG_INTERVAL

            print(
                f"Episode {games_played}/{EPISODES} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Win Rate: {recent_win_rate:.4f} | "
                f"Draw Rate: {recent_draw_rate:.4f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Q-States: {len(agent.q_table)} | "
                f"Opponents: {opponent_counts}"
            )

    agent.save("q_table.npy")
    np.save("rewards.npy", np.array(rewards, dtype=np.float64))
    np.save("win_rates.npy", np.array(win_rates, dtype=np.float64))
    np.save("draw_rates.npy", np.array(draw_rates, dtype=np.float64))
    np.save("loss_rates.npy", np.array(loss_rates, dtype=np.float64))

    return agent, rewards, win_rates, draw_rates, loss_rates


if __name__ == "__main__":
    run_training()