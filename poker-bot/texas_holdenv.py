"""
texas_holdenv.py — Texas Hold'em Environment.
Inherits from gymnasium.Env.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from hand_evaluator import compare_hands


class TexasHoldemEnv(gym.Env):
    """
    Texas Hold'em environment with 2 players.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Actions: 0 = Check / Call, 1 = Bet, 2 = Fold
        self.action_space = spaces.Discrete(3)

        # Observation Space:
        # [hole_1, hole_2, community_1, community_2, community_3, community_4, community_5,
        #  round, current_player, pot, has_active_bet, actions_this_round_count]
        # Cards encoded 0-19. -1 for empty/unknown community cards.
        # round: 0=preflop, 1=flop, 2=turn, 3=river, 4=showdown
        # current_player: 0 or 1
        # has_active_bet: 0 or 1
        self.observation_space = spaces.Box(
            low=-1,
            high=100,
            shape=(12,),
            dtype=np.int32
        )

        # Define 20-card short deck structure: 10, J, Q, K, A across 4 suits.
        self.ranks = [10, 11, 12, 13, 14]
        self.suits = ["C", "D", "H", "S"]
        self.deck_cards = [(r, s) for r in self.ranks for s in self.suits]

        # Reset variables
        self.deck = []
        self.player_hands = {0: [], 1: []}
        self.community_cards = []
        self.pot = 2
        self.round = "preflop"
        self.current_player = 0
        self.starting_player = 0
        self.has_active_bet = False
        self.bettor = None
        self.done = False
        self.actions_this_round = []

    def _create_deck(self):
        return list(self.deck_cards)

    def _shuffle_deck(self):
        self.np_random.shuffle(self.deck)

    def _card_to_id(self, card):
        if card is None:
            return -1
        return self.deck_cards.index(card)

    def _deal_cards(self):
        # Deal 2 hole cards to each player
        self.player_hands[0] = [self.deck.pop(), self.deck.pop()]
        self.player_hands[1] = [self.deck.pop(), self.deck.pop()]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = self._create_deck()
        self._shuffle_deck()

        options = options or {}
        self.starting_player = int(options.get("starting_player", 0))

        self.player_hands = {0: [], 1: []}
        self._deal_cards()

        self.community_cards = []
        self.pot = 2
        self.round = "preflop"
        self.current_player = self.starting_player
        self.has_active_bet = False
        self.bettor = None
        self.done = False
        self.actions_this_round = []

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # Player 0 hole cards
        h0 = self._card_to_id(self.player_hands[0][0])
        h1 = self._card_to_id(self.player_hands[0][1])

        # Community cards
        comm = [-1] * 5
        for i, card in enumerate(self.community_cards):
            comm[i] = self._card_to_id(card)

        # Round encoding
        round_enc = {
            "preflop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
            "showdown": 4
        }[self.round]

        obs = np.array([
            h0,
            h1,
            comm[0],
            comm[1],
            comm[2],
            comm[3],
            comm[4],
            round_enc,
            self.current_player,
            self.pot,
            1 if self.has_active_bet else 0,
            min(len(self.actions_this_round), 2)
        ], dtype=np.int32)
        return obs

    def _get_info(self, winner=None, end_reason=None):
        return {
            "round": self.round,
            "pot": self.pot,
            "current_player": self.current_player,
            "has_active_bet": self.has_active_bet,
            "winner": winner,
            "end_reason": end_reason,
            "player_hands": {
                0: [str(c) for c in self.player_hands[0]],
                1: [str(c) for c in self.player_hands[1]]
            },
            "community_cards": [str(c) for c in self.community_cards]
        }

    def get_valid_actions(self):
        if self.has_active_bet:
            return [0, 2]  # Call, Fold
        return [0, 1]      # Check, Bet

    def _get_valid_actions(self):
        return self.get_valid_actions()

    def _validate_action(self, action):
        valid = self.get_valid_actions()
        if action not in valid:
            raise ValueError(f"Action {action} is invalid. Valid: {valid}")

    def _switch_player(self):
        self.current_player = 1 - self.current_player

    def _both_players_checked(self):
        # Both players check if there's no active bet and we have had two check actions.
        # Since round always starts with player 0, if player 0 checks then player 1 checks,
        # len(actions_this_round) will be 2 and both actions are 0.
        if not self.has_active_bet and len(self.actions_this_round) == 2:
            if self.actions_this_round[0][1] == 0 and self.actions_this_round[1][1] == 0:
                return True
        return False

    def _reveal_next_community_cards(self):
        if self.round == "preflop":
            # Flop: deal 3 cards
            self.community_cards.extend([self.deck.pop(), self.deck.pop(), self.deck.pop()])
            self.round = "flop"
        elif self.round == "flop":
            # Turn: deal 1 card
            self.community_cards.append(self.deck.pop())
            self.round = "turn"
        elif self.round == "turn":
            # River: deal 1 card
            self.community_cards.append(self.deck.pop())
            self.round = "river"
        elif self.round == "river":
            self.round = "showdown"

    def _showdown(self):
        # Compare hands and return winner (0, 1, or None for draw)
        winner = compare_hands(
            self.player_hands[0] + self.community_cards,
            self.player_hands[1] + self.community_cards
        )
        self.done = True
        return winner

    def _get_reward(self, winner):
        if winner == 0:
            return self.pot
        elif winner == 1:
            return -self.pot
        else:
            return 0

    def _finish_round_or_game(self):
        self.actions_this_round = []
        if self.round == "river":
            # Transition to showdown
            self.round = "showdown"
            winner = self._showdown()
            reward = self._get_reward(winner)
            return reward, True, winner
        else:
            self._reveal_next_community_cards()
            self.current_player = self.starting_player
            return 0, False, None

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot step in a finished episode. Reset the environment.")

        # Ensure action is valid
        self._validate_action(action)

        acting_player = self.current_player
        self.actions_this_round.append((acting_player, action))

        # Handle fold immediately
        if action == 2:
            # Fold: opponent wins
            winner = 1 - acting_player
            self.done = True
            reward = self._get_reward(winner)
            return self._get_obs(), reward, True, False, self._get_info(winner=winner, end_reason="fold")

        # Handle bet
        if action == 1:
            self.pot += 1
            self.has_active_bet = True
            self.bettor = acting_player
            self._switch_player()
            return self._get_obs(), 0, False, False, self._get_info()

        # Handle Check / Call (action == 0)
        if self.has_active_bet:
            # This is a Call
            self.pot += 1
            self.has_active_bet = False
            self.bettor = None
            # Betting round ends immediately on Call
            reward, term, winner = self._finish_round_or_game()
            end_reason = "showdown" if term else None
            return self._get_obs(), reward, term, False, self._get_info(winner=winner, end_reason=end_reason)
        else:
            # This is a Check
            if len(self.actions_this_round) == 2:
                # Both checked
                reward, term, winner = self._finish_round_or_game()
                end_reason = "showdown" if term else None
                return self._get_obs(), reward, term, False, self._get_info(winner=winner, end_reason=end_reason)
            else:
                self._switch_player()
                return self._get_obs(), 0, False, False, self._get_info()


# Aliases for compatibility
SimplifiedTexasHoldemEnv = TexasHoldemEnv
KuhnPokerEnv = TexasHoldemEnv