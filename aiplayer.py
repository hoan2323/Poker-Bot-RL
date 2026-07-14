"""
aiplayer.py - AI Player Interface
Load trained model and play against it
"""

import torch
import numpy as np

from nfsp_agent import NSFPAgent
from environment import ShortDeckPokerEnv
from config import DEVICE


class AIPlayer:
    """
    Interface for playing against trained NFSP agent
    """

    def __init__(self, model_path=None):
        self.agent = None
        self.env = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

    def load_model(self, path):
        """Load trained NFSP agent"""
        self.agent = NSFPAgent()
        self.agent.load(path)
        self.agent.q_net.eval()
        self.agent.policy_net.eval()
        self.model_path = path
        print(f"Loaded model from {path}")

    def set_evaluate_mode(self):
        """Set agent to evaluation mode (no exploration)"""
        if self.agent:
            self.agent.eta = 0.0

    def set_training_mode(self):
        """Set agent back to training mode"""
        if self.agent:
            from config import ANTICIPATORY_PARAM
            self.agent.eta = ANTICIPATORY_PARAM

    def get_action(self, state, valid_actions=None):
        """
        Get action from agent

        Args:
            state: state vector (186 features)
            valid_actions: list of valid actions (optional)

        Returns:
            action: chosen action
        """
        if self.agent is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        action = self.agent.choose_action(state, valid_actions, evaluate=True)

        # Ensure valid action
        if valid_actions is not None and action not in valid_actions:
            action = valid_actions[0]

        return action

    def start_new_game(self, starting_player=0):
        """Start a new game"""
        self.env = ShortDeckPokerEnv()
        self.env.reset(starting_player=starting_player)
        return self.env.get_state(0)

    def play(self, action):
        """
        Play action in current game

        Args:
            action: action to take

        Returns:
            (next_state, reward, done, info)
        """
        if self.env is None:
            raise RuntimeError("Game not started. Call start_new_game() first.")

        return self.env.step(action)

    def get_game_state(self):
        """Get current game state"""
        if self.env is None:
            return None
        return {
            'state': self.env.get_state(0),
            'valid_actions': self.env.get_valid_actions(),
            'pot': self.env.pot,
            'round': self.env.round,
            'board': self.env.board,
            'player_hand': self.env.hands[0],
            'opponent_hand': self.env.hands[1],  # Hidden for real game
            'done': self.env.done,
            'winner': self.env.winner
        }

    def is_game_over(self):
        """Check if game is over"""
        return self.env.done if self.env else False


def play_game_vs_ai(ai_player, human_player=0, verbose=True):
    """
    Play a single game against AI

    Args:
        ai_player: AIPlayer instance
        human_player: which player is human (0 or 1)
        verbose: print game progress

    Returns:
        winner: 0, 1, or None
    """
    # Start game
    state = ai_player.start_new_game(starting_player=human_player)

    if verbose:
        print("\n" + "=" * 50)
        print("NEW GAME")
        print("=" * 50)

    while not ai_player.is_game_over():
        game_state = ai_player.get_game_state()

        if verbose:
            print(f"\nPot: {game_state['pot']} | Round: {game_state['round']}")
            print(f"Board: {[c for c in game_state['board']]}")

            if game_state['player_hand']:
                from environment import card_to_string
                print(f"Your hand: {[card_to_string(c) for c in game_state['player_hand']]}")

        current = ai_player.env.current_player

        if current == human_player:
            # Human's turn
            if verbose:
                print(f"Your turn!")
                print(f"Valid actions: {game_state['valid_actions']}")

            # Get human input
            action = int(input("Your action: "))
        else:
            # AI's turn
            if verbose:
                print("AI is thinking...")

            action = ai_player.get_action(state, game_state['valid_actions'])

            if verbose:
                action_names = {0: 'Check/Call', 1: 'Bet/Raise', 2: 'Fold'}
                print(f"AI action: {action_names.get(action, action)}")

        # Play
        state, reward, done, info = ai_player.play(action)

    # Game over
    if verbose:
        print("\n" + "=" * 50)
        if ai_player.env.winner == human_player:
            print("YOU WIN!")
        elif ai_player.env.winner == 1 - human_player:
            print("AI WINS!")
        else:
            print("TIE!")

        print(f"Pot: {ai_player.env.pot}")

    return ai_player.env.winner


if __name__ == "__main__":
    # Test AI player interface
    print("Testing AI Player Interface...")

    # Create AI player (without loading model)
    ai = AIPlayer()

    # Start game
    state = ai.start_new_game()

    print(f"Initial state shape: {state.shape}")
    print(f"Valid actions: {ai.env.get_valid_actions()}")

    # Get AI action
    action = ai.get_action(state)
    print(f"AI would play: {action}")

    print("\nAI Player interface working!")
