import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import Counter


class TexasHoldemEnv(gym.Env):
    """Simplified Texas Hold'em environment for RL."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1,
            high=51,
            shape=(7,),
            dtype=np.int32
        )

        self.ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.suits = ["C", "D", "H", "S"]

        self.deck = []
        self.player_hands = [[], []]
        self.community_cards = []
        self.pot = 0
        self.phase = "preflop"
        self.current_player = 0
        self.done = False
        self.round_actions = []

    def _create_deck(self):
        """Create a standard 52-card deck."""
        deck = []
        for rank in self.ranks:
            for suit in self.suits:
                deck.append((rank, suit))
        return deck

    def _card_to_id(self, card):
        """Convert (rank, suit) to integer 0-51."""
        rank, suit = card
        rank_idx = self.ranks.index(rank)
        suit_idx = self.suits.index(suit)
        return rank_idx * 4 + suit_idx

    def _get_obs(self):
        """Get observation for Player 0."""
        hole_1 = self._card_to_id(self.player_hands[0][0])
        hole_2 = self._card_to_id(self.player_hands[0][1])

        community = [-1, -1, -1]
        for i, card in enumerate(self.community_cards):
            community[i] = self._card_to_id(card)

        phase_val = 0
        if self.phase == "preflop":
            phase_val = 0
        elif self.phase == "flop":
            phase_val = 1
        elif self.phase == "showdown":
            phase_val = 2

        obs = np.array(
            [hole_1, hole_2, community[0], community[1], community[2],
             phase_val, self.current_player],
            dtype=np.int32
        )
        return obs

    def reset(self, seed=None, options=None):
        """Reset environment for new game."""
        super().reset(seed=seed)

        self.deck = self._create_deck()
        self.np_random.shuffle(self.deck)

        self.player_hands = [[], []]
        self.player_hands[0] = [self.deck.pop(), self.deck.pop()]
        self.player_hands[1] = [self.deck.pop(), self.deck.pop()]

        self.community_cards = []
        self.pot = 2
        self.phase = "preflop"
        self.current_player = 0
        self.done = False
        self.round_actions = []

        state = self._get_obs()
        info = {}
        return state, info

    def step(self, action):
        """Execute one action. Returns (state, reward, terminated, truncated, info)."""
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")

        if action not in [0, 1, 2]:
            raise ValueError("Invalid action. Must be 0=Check, 1=Bet, 2=Fold.")

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Fold
        if action == 2:
            self.done = True
            terminated = True
            if self.current_player == 0:
                # Player 0 folds -> Player 1 wins
                winner = 1
                reward = -2
            else:
                # Player 1 folds -> Player 0 wins
                winner = 0
                reward = 2
            info["winner"] = winner
            state = self._get_obs()
            return state, reward, terminated, truncated, info

        # Check or Bet
        if action == 1:
            self.pot += 1

        self.round_actions.append((self.current_player, action))

        # Check if both players acted this round
        if len(self.round_actions) == 2:
            if self.phase == "preflop":
                self._reveal_flop()
                self.phase = "flop"
                self.current_player = 0
                self.round_actions = []
            elif self.phase == "flop":
                self.phase = "showdown"
                reward, winner = self._resolve_showdown()
                self.done = True
                terminated = True
                info["winner"] = winner
        else:
            # Switch to other player
            self.current_player = 1 - self.current_player

        state = self._get_obs()
        return state, reward, terminated, truncated, info

    def _reveal_flop(self):
        """Deal 3 community cards."""
        self.community_cards = [self.deck.pop() for _ in range(3)]

    def _resolve_showdown(self):
        """Compare hands and return (reward, winner)."""
        hand0 = self.player_hands[0] + self.community_cards
        hand1 = self.player_hands[1] + self.community_cards

        score0 = self._evaluate_hand(hand0)
        score1 = self._evaluate_hand(hand1)

        if score0 > score1:
            return 2, 0
        elif score1 > score0:
            return -2, 1
        else:
            return 0, None

    def _evaluate_hand(self, cards):
        """
        Simple hand evaluator for 5 cards.
        Returns a tuple for comparison: (hand_rank, tiebreakers...).

        Hand rankings:
        8 - Straight Flush
        7 - Four of a Kind
        6 - Full House
        5 - Flush
        4 - Straight
        3 - Three of a Kind
        2 - Two Pair
        1 - One Pair
        0 - High Card
        """
        ranks = sorted([c[0] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        rank_counts = Counter(ranks)

        counts = sorted(rank_counts.values(), reverse=True)
        # Ranks sorted by frequency then by rank value
        ranks_by_freq = sorted(rank_counts.keys(),
                               key=lambda r: (rank_counts[r], r), reverse=True)

        is_flush = len(set(suits)) == 1

        # Check straight
        sorted_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = 0
        if len(sorted_ranks) == 5:
            if sorted_ranks[-1] - sorted_ranks[0] == 4:
                is_straight = True
                straight_high = sorted_ranks[-1]
            # Ace-low straight: A-2-3-4-5
            if sorted_ranks == [2, 3, 4, 5, 14]:
                is_straight = True
                straight_high = 5

        if is_straight and is_flush:
            return (8, straight_high)
        if counts == [4, 1]:
            return (7, ranks_by_freq[0], ranks_by_freq[1])
        if counts == [3, 2]:
            return (6, ranks_by_freq[0], ranks_by_freq[1])
        if is_flush:
            return (5,) + tuple(ranks)
        if is_straight:
            return (4, straight_high)
        if counts == [3, 1, 1]:
            return (3, ranks_by_freq[0], ranks_by_freq[1], ranks_by_freq[2])
        if counts == [2, 2, 1]:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return (2, pairs[0], pairs[1], kicker)
        if counts == [2, 1, 1, 1]:
            pair = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return (1, pair) + tuple(kickers)
        return (0,) + tuple(ranks)


# Alias for compatibility
KuhnPokerEnv = TexasHoldemEnv