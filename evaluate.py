"""
evaluate.py - Evaluation for NFSP Poker Bot
Evaluate trained agents against baseline opponents
"""

import numpy as np
import random

from environment import ShortDeckPokerEnv


# Baseline Opponents
def random_opponent(env):
    """Random opponent - chooses randomly"""
    actions = env.get_valid_actions()
    return random.choice(actions)


def always_check_or_call(env):
    """Always check if possible, call if facing bet"""
    actions = env.get_valid_actions()
    if 0 in actions:
        return 0  # Check/Call
    return actions[0]


def always_bet_or_call(env):
    """Always bet if possible, call if facing bet"""
    actions = env.get_valid_actions()
    if 1 in actions:
        return 1  # Bet/Raise
    return 0  # Call


def call_station(env):
    """Call station - never folds"""
    actions = env.get_valid_actions()
    if 0 in actions:
        return 0  # Always call/check
    return actions[0]


def tight_agent(env):
    """Tight agent - only plays strong hands"""
    from environment import evaluate_hand, get_rank_counts

    player = env.current_player
    hand = env.hands[player]
    board = env.board
    all_cards = hand + board

    hand_result = evaluate_hand(all_cards)
    hand_rank = hand_result[0] if hand_result else 0

    actions = env.get_valid_actions()

    # If facing bet, only call with strong hand
    if 2 in actions and env.bet_size > 0:
        if hand_rank < 2:  # Less than two pair
            return 2  # Fold

    # If can bet, only bet with strong hand
    if 1 in actions and hand_rank >= 2:
        return 1  # Bet

    return 0  # Check/Call


def aggressive_agent(env):
    """Aggressive agent - bets often"""
    actions = env.get_valid_actions()
    if 1 in actions:
        return 1  # Bet/Raise
    return 0  # Call


def heuristic_agent(env):
    """Heuristic agent using hand strength"""
    from environment import evaluate_hand

    player = env.current_player
    hand = env.hands[player]
    board = env.board
    all_cards = hand + board

    hand_result = evaluate_hand(all_cards)
    hand_rank = hand_result[0] if hand_result else 0

    actions = env.get_valid_actions()

    # Strong hand (two pair+)
    if hand_rank >= 2:
        if 1 in actions:
            return 1  # Bet
        return 0  # Call

    # Medium hand (one pair)
    if hand_rank == 1:
        if 1 in actions and random.random() < 0.5:
            return 1  # Sometimes bet
        if 0 in actions:
            return 0  # Call
        return 2  # Fold

    # Weak hand
    if 2 in actions:
        return 2  # Fold
    if 1 in actions and random.random() < 0.1:
        return 1  # Occasional bluff
    return 0  # Check


# Opponent registry
OPPONENTS = {
    'random': random_opponent,
    'always_check': always_check_or_call,
    'always_bet': always_bet_or_call,
    'call_station': call_station,
    'tight': tight_agent,
    'aggressive': aggressive_agent,
    'heuristic': heuristic_agent
}


def evaluate_agent(agent, opponent_name='random', n_games=5000, verbose=True):
    """
    Evaluate agent against a specific opponent

    Args:
        agent: NSFPAgent
        opponent_name: name of opponent
        n_games: number of games to play
        verbose: print results

    Returns:
        win_rate: fraction of games won
    """
    if opponent_name not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent_name}")

    opponent = OPPONENTS[opponent_name]
    env = ShortDeckPokerEnv()

    wins = 0
    losses = 0
    ties = 0
    total_reward = 0

    for _ in range(n_games):
        # Reset game
        state = env.reset(starting_player=random.randint(0, 1))

        # Alternate who goes first
        agent_player = env.current_player

        # Play game
        while not env.done:
            current_player = env.current_player

            if current_player == agent_player:
                # Agent's turn
                valid_actions = env.get_valid_actions()
                action = agent.choose_action(state, evaluate=True)
                if action not in valid_actions:
                    action = valid_actions[0]
            else:
                # Opponent's turn
                action = opponent(env)

            # Execute
            next_state, reward, done, info = env.step(action)

            state = next_state

        # Record result
        if env.winner == agent_player:
            wins += 1
            total_reward += env.pot
        elif env.winner == 1 - agent_player:
            losses += 1
            total_reward -= env.pot
        else:
            ties += 1

    total = wins + losses + ties
    win_rate = wins / total if total > 0 else 0

    if verbose:
        print(f"\n--- Evaluation: Agent vs {opponent_name} ---")
        print(f"Games: {total}")
        print(f"Wins: {wins} ({win_rate:.2%})")
        print(f"Losses: {losses} ({-losses/total:.2%})")
        print(f"Ties: {ties} ({ties/total:.2%})")
        print(f"Average Reward: {total_reward/total:.2f}")

    return win_rate


def evaluate_all(agent, n_games=2000, verbose=True):
    """
    Evaluate agent against all opponents

    Returns:
        dict of opponent_name -> win_rate
    """
    results = {}

    opponent_names = ['random', 'call_station', 'tight', 'aggressive', 'heuristic']

    for opponent_name in opponent_names:
        results[opponent_name] = evaluate_agent(
            agent, opponent_name, n_games=n_games, verbose=verbose
        )

    return results


def evaluate_self_play(agent1, agent2, n_games=5000, verbose=True):
    """
    Evaluate two agents playing against each other
    """
    env = ShortDeckPokerEnv()

    wins1 = 0
    wins2 = 0
    ties = 0

    for _ in range(n_games):
        state = env.reset(starting_player=random.randint(0, 1))

        while not env.done:
            current_player = env.current_player

            if current_player == 0:
                action = agent1.choose_action(state, evaluate=True)
            else:
                action = agent2.choose_action(state, evaluate=True)

            valid_actions = env.get_valid_actions()
            if action not in valid_actions:
                action = valid_actions[0]

            next_state, reward, done, info = env.step(action)
            state = next_state

        if env.winner == 0:
            wins1 += 1
        elif env.winner == 1:
            wins2 += 1
        else:
            ties += 1

    total = wins1 + wins2 + ties
    win_rate_1 = wins1 / total if total > 0 else 0
    win_rate_2 = wins2 / total if total > 0 else 0

    if verbose:
        print(f"\n--- Self-Play Evaluation ---")
        print(f"Games: {total}")
        print(f"Agent 1 Wins: {wins1} ({win_rate_1:.2%})")
        print(f"Agent 2 Wins: {wins2} ({win_rate_2:.2%})")
        print(f"Ties: {ties} ({ties/total:.2%})")

    return win_rate_1, win_rate_2


if __name__ == "__main__":
    # Test evaluation
    from nfsp_agent import NSFPAgent

    print("Testing Evaluation...")

    # Create random agent for testing
    agent = NSFPAgent()

    # Quick evaluation
    print("\n--- Quick Evaluation (100 games each) ---")
    evaluate_agent(agent, 'random', n_games=100, verbose=True)
    evaluate_agent(agent, 'heuristic', n_games=100, verbose=True)
