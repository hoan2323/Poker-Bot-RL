"""
train.py - NFSP Training Loop
"""

import sys
import torch
import numpy as np
import random
import time
from collections import deque

print("Loading config...", flush=True)
from config import (
    TRAIN_ITERATIONS, RANDOM_SEED, TARGET_UPDATE_FREQ,
    MIN_RL_SAMPLES, MIN_SL_SAMPLES, DEVICE
)
print(f"Config loaded. Device: {DEVICE}", flush=True)

from environment import ShortDeckPokerEnv
from nfsp_agent import NSFPAgent

print("Modules imported.", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def play_episode(env, agent1, agent2):
    """Play one episode"""
    state = env.reset(starting_player=random.randint(0, 1))

    while not env.done:
        current_player = env.current_player
        agent = agent1 if current_player == 0 else agent2
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)

        if action not in valid_actions:
            action = valid_actions[0]

        next_state, reward, done, info = env.step(action)

        if current_player == 0:
            agent1.store_experience(state, action, reward, next_state, done, valid_actions)
        else:
            agent2.store_experience(state, action, -reward, next_state, done, valid_actions)

        state = next_state

    return env.winner


def train(n_iterations=TRAIN_ITERATIONS):
    print("=" * 60, flush=True)
    print("NFSP Poker Bot Training", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Iterations: {n_iterations:,}", flush=True)

    set_seed(RANDOM_SEED)

    print("Creating environment...", flush=True)
    env = ShortDeckPokerEnv()

    print("Creating agents...", flush=True)
    agent1 = NSFPAgent()
    agent2 = NSFPAgent()

    recent_wins = deque(maxlen=1000)
    start_time = time.time()
    steps_since_sync = 0

    print(f"\nStarting training loop... (press Ctrl+C to stop)", flush=True)
    print("-" * 60, flush=True)

    for iteration in range(n_iterations):
        winner = play_episode(env, agent1, agent2)

        if winner == 0:
            recent_wins.append(1)
        elif winner == 1:
            recent_wins.append(0)
        else:
            recent_wins.append(0.5)

        if (iteration + 1) % 20 == 0:
            agent1.update()
            agent2.update()

        steps_since_sync += 1
        if steps_since_sync >= TARGET_UPDATE_FREQ:
            agent1.sync_target_network()
            agent2.sync_target_network()
            steps_since_sync = 0

        if (iteration + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            speed = (iteration + 1) / elapsed if elapsed > 0 else 0
            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.5
            print(f"Iter {iteration+1:,} | Win: {win_rate:.1%} | "
                  f"Speed: {speed:.0f}/s | RL: {len(agent1.rl_buffer):,} | "
                  f"SL: {len(agent1.sl_reservoir):,}", flush=True)

    print("-" * 60, flush=True)
    print("Training complete!", flush=True)

    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"Final win rate: {sum(recent_wins)/len(recent_wins):.1%}", flush=True)

    print("Saving models...", flush=True)
    agent1.save("nfsp_agent_final.pt")
    agent2.save("nfsp_agent2_final.pt")

    return agent1, agent2


if __name__ == "__main__":
    print("\n" + "=" * 60, flush=True)
    print("Starting NFSP Training", flush=True)
    print("=" * 60 + "\n", flush=True)

    try:
        agent1, agent2 = train(n_iterations=TRAIN_ITERATIONS)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.", flush=True)
        print("Models not saved.", flush=True)
    except Exception as e:
        print(f"\n\nERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
