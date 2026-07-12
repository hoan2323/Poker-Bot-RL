"""
train.py - NFSP Training Loop
Self-play training for poker bot
"""

import torch
import numpy as np
import random
import time
from collections import deque

from config import (
    TRAIN_ITERATIONS, UPDATE_FREQUENCY, EVAL_FREQUENCY,
    RANDOM_SEED
)
from environment import ShortDeckPokerEnv
from nfsp_agent import NSFPAgent
from evaluate import evaluate_agent


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def play_episode(env, agent1, agent2, store_experiences=True):
    """
    Play one episode of self-play between two agents

    Returns:
        - winner: 0, 1, or None (tie)
        - stats: dict of episode statistics
    """
    state = env.reset(starting_player=random.randint(0, 1))

    episode_stats = {
        'actions_p1': 0,
        'actions_p2': 0,
        'pot_size': 0
    }

    while not env.done:
        current_player = env.current_player
        agent = agent1 if current_player == 0 else agent2

        # Get valid actions
        valid_actions = env.get_valid_actions()

        # Choose action
        action = agent.choose_action(state)

        # Ensure valid action
        if action not in valid_actions:
            action = valid_actions[0]

        # Store action count
        if current_player == 0:
            episode_stats['actions_p1'] += 1
        else:
            episode_stats['actions_p2'] += 1

        # Execute action
        next_state, reward, done, info = env.step(action)

        # Store experience for the acting player
        if store_experiences:
            # Determine opponent's perspective reward
            opponent_reward = -reward if current_player == 0 else -reward

            if current_player == 0:
                agent1.store_experience(state, action, reward, next_state, done)
            else:
                agent2.store_experience(state, action, opponent_reward, next_state, done)

        state = next_state

    episode_stats['pot_size'] = env.pot
    episode_stats['winner'] = env.winner

    return env.winner, episode_stats


def update_agents(agent1, agent2):
    """Update both agents"""
    rl_losses_1, sl_losses_1 = [], []
    rl_losses_2, sl_losses_2 = [], []

    # Update agent 1
    rl_loss_1 = agent1.update_rl()
    sl_loss_1 = agent1.update_sl()
    if rl_loss_1 is not None:
        rl_losses_1.append(rl_loss_1)
    if sl_loss_1 is not None:
        sl_losses_1.append(sl_loss_1)

    # Update agent 2
    rl_loss_2 = agent2.update_rl()
    sl_loss_2 = agent2.update_sl()
    if rl_loss_2 is not None:
        rl_losses_2.append(rl_loss_2)
    if sl_loss_2 is not None:
        sl_losses_2.append(sl_loss_2)

    # Sync target networks periodically
    agent1.sync_target_network()
    agent2.sync_target_network()

    return {
        'agent1_rl_loss': np.mean(rl_losses_1) if rl_losses_1 else None,
        'agent1_sl_loss': np.mean(sl_losses_1) if sl_losses_1 else None,
        'agent2_rl_loss': np.mean(rl_losses_2) if rl_losses_2 else None,
        'agent2_sl_loss': np.mean(sl_losses_2) if sl_losses_2 else None,
    }


def log_progress(iteration, stats, eval_results=None):
    """Log training progress"""
    agent_stats = stats['agent'].get_statistics()

    log_str = f"\nIteration {iteration}/{TRAIN_ITERATIONS}"
    log_str += f"\n  Steps: {agent_stats['total_steps']}"
    log_str += f"\n  RL Buffer: {agent_stats['rl_buffer_size']}"
    log_str += f"\n  SL Reservoir: {agent_stats['sl_reservoir_size']}"
    log_str += f"\n  RL Updates: {agent_stats['rl_updates']}"
    log_str += f"\n  SL Updates: {agent_stats['sl_updates']}"

    if stats.get('rl_loss'):
        log_str += f"\n  RL Loss: {stats['rl_loss']:.4f}"
    if stats.get('sl_loss'):
        log_str += f"\n  SL Loss: {stats['sl_loss']:.4f}"

    if eval_results:
        log_str += f"\n  --- Evaluation ---"
        log_str += f"\n  vs Random: {eval_results['random']:.2%}"
        log_str += f"\n  vs Heuristic: {eval_results['heuristic']:.2%}"
        log_str += f"\n  Self-play: {eval_results['self_play']:.2%}"

    print(log_str)


def train(n_iterations=TRAIN_ITERATIONS, eval_every=EVAL_FREQUENCY):
    """Main training loop"""
    print("=" * 60)
    print("NFSP Poker Bot Training")
    print("=" * 60)

    # Set seed
    set_seed(RANDOM_SEED)

    # Create environment
    env = ShortDeckPokerEnv()

    # Create agents
    print("\nInitializing agents...")
    agent1 = NSFPAgent()
    agent2 = NSFPAgent()

    # Training stats
    recent_wins = deque(maxlen=1000)
    start_time = time.time()

    print(f"\nTraining for {n_iterations} iterations...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for iteration in range(n_iterations):
        # Play episode
        winner, episode_stats = play_episode(env, agent1, agent2)

        # Track wins
        if winner == 0:
            recent_wins.append(1)
        elif winner == 1:
            recent_wins.append(0)
        else:
            recent_wins.append(0.5)  # Tie counts as half

        # Update agents
        update_stats = update_agents(agent1, agent2)

        # Periodic evaluation
        if (iteration + 1) % eval_every == 0:
            eval_results = {
                'random': evaluate_agent(agent1, 'random', n_games=1000),
                'heuristic': evaluate_agent(agent1, 'heuristic', n_games=1000),
                'self_play': sum(recent_wins) / len(recent_wins) if recent_wins else 0.5
            }

            stats = {
                'agent': agent1,
                'rl_loss': update_stats.get('agent1_rl_loss'),
                'sl_loss': update_stats.get('agent1_sl_loss')
            }

            log_progress(iteration + 1, stats, eval_results)

            # Save checkpoint
            checkpoint_path = f"checkpoint_iter_{iteration + 1}.pt"
            agent1.save(checkpoint_path)

    # Save final models
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/3600:.1f} hours)")
    print(f"Final win rate: {sum(recent_wins)/len(recent_wins):.2%}")

    agent1.save("nfsp_agent_final.pt")
    agent2.save("nfsp_agent2_final.pt")

    return agent1, agent2


if __name__ == "__main__":
    agent1, agent2 = train(n_iterations=100000, eval_every=10000)
