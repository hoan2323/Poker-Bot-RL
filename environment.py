"""
environment.py - Short Deck Poker Environment
20 cards: 10, J, Q, K, A x 4 suits
4 betting rounds: preflop, flop, turn, river
"""

import numpy as np
import random


class ShortDeckPokerEnv:
    """
    Short Deck Texas Hold'em Environment for NFSP
    """

    def __init__(self):
        # Card representation: 0-19
        # Rank: card // 4 (0=10, 1=J, 2=Q, 3=K, 4=A)
        # Suit: card % 4 (0=C, 1=D, 2=H, 3=S)

        self.num_players = 2
        self.deck_size = 20
        self.hole_cards = 2
        self.community_cards = 5

        # Game state
        self.deck = []
        self.hands = {0: [], 1: []}
        self.board = []
        self.pot = 2
        self.round = 0  # 0=preflop, 1=flop, 2=turn, 3=river, 4=showdown
        self.current_player = 0
        self.starting_player = 0

        # Betting state
        self.bet_size = 0  # Current bet in this round
        self.player_bets = {0: 0, 1: 0}  # Total bet this round
        self.actions_this_round = []

        # Terminal state
        self.done = False
        self.winner = None
        self.rewards = {0: 0, 1: 0}

    def reset(self, starting_player=0):
        """Reset game to initial state"""
        self.deck = list(range(20))
        random.shuffle(self.deck)

        # Deal hole cards
        self.hands[0] = [self.deck.pop(), self.deck.pop()]
        self.hands[1] = [self.deck.pop(), self.deck.pop()]

        # Reset game state
        self.board = []
        self.pot = 2
        self.round = 0
        self.starting_player = starting_player
        self.current_player = starting_player

        # Reset betting
        self.bet_size = 0
        self.player_bets = {0: 0, 1: 0}
        self.actions_this_round = []

        # Reset terminal
        self.done = False
        self.winner = None
        self.rewards = {0: 0, 1: 0}

        return self.get_state(0)

    def get_state(self, player):
        """
        Get 186-bit state vector for player
        """
        state = np.zeros(186, dtype=np.float32)

        # Player's hole cards (one-hot encoding)
        for i, card in enumerate(self.hands[player]):
            state[card] = 1.0

        # Opponent's hole cards (zeros - hidden)

        # Community cards (one-hot encoding)
        for i, card in enumerate(self.board):
            state[80 + card] = 1.0

        # Additional features (normalized)
        state[180] = min(self.pot, 20) / 20.0  # Pot size (normalized)
        state[181] = self.round / 4.0  # Round index
        state[182] = 1.0 if self.current_player == player else 0.0  # Is turn
        state[183] = 1.0 if self.bet_size > 0 else 0.0  # Has active bet
        state[184] = self.bet_size / 5.0  # Bet size (normalized)
        state[185] = self.player_bets[player] / 5.0  # Player's bet

        return state

    def get_valid_actions(self):
        """Get valid actions for current player"""
        if self.done:
            return []

        if self.bet_size == 0:
            # No bet: Check or Bet
            return [0, 1]
        else:
            # Facing bet: Call, Raise, or Fold
            return [0, 1, 2]

    def step(self, action):
        """
        Execute action for current player
        Returns: (state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Game is over. Call reset() first.")

        valid = self.get_valid_actions()
        if action not in valid:
            raise ValueError(f"Invalid action {action}. Valid: {valid}")

        player = self.current_player
        self.actions_this_round.append((player, action))

        reward = 0

        # Handle Fold
        if action == 2:
            self.done = True
            self.winner = 1 - player
            self.rewards[player] = -self.pot
            self.rewards[1 - player] = self.pot
            reward = self.rewards[0]
            return self.get_state(0), reward, True, self._get_info()

        # Handle Bet/Check
        if self.bet_size == 0:
            if action == 0:
                # Check
                if len(self.actions_this_round) == 2:
                    self._advance_round()
                else:
                    self.current_player = 1 - self.current_player
            else:
                # Bet
                self.bet_size = 1
                self.player_bets[player] = 1
                self.pot += 1
                self.current_player = 1 - self.current_player

        else:
            # Facing bet
            if action == 0:
                # Call
                to_call = self.bet_size - self.player_bets[player]
                self.player_bets[player] = self.bet_size
                self.pot += to_call
                self.bet_size = 0
                self._advance_round()

            else:
                # Raise
                self.bet_size += 1
                self.player_bets[player] = self.bet_size
                self.pot += 1
                self.current_player = 1 - self.current_player

        return self.get_state(0), reward, self.done, self._get_info()

    def _advance_round(self):
        """Advance to next round or showdown"""
        self.actions_this_round = []
        self.player_bets = {0: 0, 1: 0}

        if self.round == 0:
            # Preflop -> Flop
            for _ in range(3):
                self.board.append(self.deck.pop())
            self.round = 1

        elif self.round == 1:
            # Flop -> Turn
            self.board.append(self.deck.pop())
            self.round = 2

        elif self.round == 2:
            # Turn -> River
            self.board.append(self.deck.pop())
            self.round = 3

        elif self.round == 3:
            # River -> Showdown
            self.round = 4
            self._showdown()

    def _showdown(self):
        """Compare hands at showdown"""
        self.done = True
        winner = compare_hands(
            self.hands[0] + self.board,
            self.hands[1] + self.board
        )
        self.winner = winner

        if winner == 1:  # Player 1 wins
            self.rewards = {0: -self.pot, 1: self.pot}
        elif winner == -1:  # Player 0 wins
            self.rewards = {0: self.pot, 1: -self.pot}
        else:  # Tie
            self.rewards = {0: 0, 1: 0}

    def get_reward(self, player):
        """Get reward for player"""
        return self.rewards.get(player, 0)

    def _get_info(self):
        """Get info dict"""
        return {
            "round": self.round,
            "pot": self.pot,
            "current_player": self.current_player,
            "board": self.board,
            "hands": {p: self.hands[p] for p in [0, 1]},
            "winner": self.winner,
            "done": self.done
        }


def card_to_rank(card):
    """Get rank from card index (0=10, 1=J, 2=Q, 3=K, 4=A)"""
    return card // 4


def card_to_suit(card):
    """Get suit from card index (0=C, 1=D, 2=H, 3=S)"""
    return card % 4


def card_to_string(card):
    """Convert card index to string"""
    ranks = ['10', 'J', 'Q', 'K', 'A']
    suits = ['C', 'D', 'H', 'S']
    return ranks[card_to_rank(card)] + suits[card_to_suit(card)]


# Hand evaluation functions
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


def get_rank_counts(cards):
    """Get count of cards for each rank"""
    counts = [0] * 5
    for card in cards:
        if 0 <= card < 20:
            rank = card // 4
            counts[rank] += 1
    return counts


def get_suit_counts(cards):
    """Get count of cards for each suit"""
    counts = [0] * 4
    for card in cards:
        if 0 <= card < 20:
            suit = card % 4
            counts[suit] += 1
    return counts


def is_flush(cards):
    """Check if 5+ cards form a flush"""
    return any(s >= 5 for s in get_suit_counts(cards))


def is_straight(cards):
    """
    Check if these 5 cards form a straight.
    Short deck: only 10-J-Q-K-A (ranks 0,1,2,3,4) is valid.
    """
    if len(cards) != 5:
        return False

    # Get rank of each card (each rank = 1 card in short deck)
    ranks = set()
    for card in cards:
        rank = card // 4
        # Already have this rank? Not a straight!
        if rank in ranks:
            return False
        ranks.add(rank)

    # Check if we have exactly {0, 1, 2, 3, 4} (10,J,Q,K,A)
    return ranks == {0, 1, 2, 3, 4}


def evaluate_hand(cards):
    """
    Evaluate best 5-card hand from 7 cards
    Returns: (hand_rank, kicker tuple)
    """
    from itertools import combinations

    if len(cards) < 5:
        return (HIGH_CARD, ())

    best = None

    for combo in combinations(cards, 5):
        hand = evaluate_5_cards(list(combo))
        if best is None or hand > best:
            best = hand

    return best


def evaluate_5_cards(cards):
    """Evaluate a 5-card hand"""
    rank_counts = get_rank_counts(cards)
    has_flush = is_flush(cards)
    has_straight = is_straight(cards)

    # Straight flush
    if has_flush and has_straight:
        return (STRAIGHT_FLUSH, ())

    # Four of a kind
    if 4 in rank_counts:
        kicker = [r for r in range(4, -1, -1) if rank_counts[r] != 4][0]
        return (FOUR_OF_A_KIND, (rank_counts.index(4), kicker))

    # Full house
    if 3 in rank_counts and 2 in rank_counts:
        trips = rank_counts.index(3)
        pair = [r for r in range(4, -1, -1) if rank_counts[r] == 2][0]
        return (FULL_HOUSE, (trips, pair))

    # Flush
    if has_flush:
        kickers = sorted([c // 4 for c in cards], reverse=True)[:5]
        return (FLUSH, tuple(kickers))

    # Straight
    if has_straight:
        return (STRAIGHT, ())

    # Three of a kind
    if 3 in rank_counts:
        trips = rank_counts.index(3)
        kickers = tuple(r for r in range(4, -1, -1) if r != trips)
        return (THREE_OF_A_KIND, kickers)

    # Two pair
    pairs = sorted([r for r in range(5) if rank_counts[r] == 2], reverse=True)
    if len(pairs) >= 2:
        kicker = [r for r in range(4, -1, -1) if r not in pairs][0]
        return (TWO_PAIR, (pairs[0], pairs[1], kicker))

    # One pair
    if 2 in rank_counts:
        pair = rank_counts.index(2)
        kickers = tuple(r for r in range(4, -1, -1) if r != pair)
        return (ONE_PAIR, kickers)

    # High card
    kickers = tuple(sorted([c // 4 for c in cards], reverse=True))
    return (HIGH_CARD, kickers)


def compare_hands(cards1, cards2):
    """
    Compare two hands (each is list of 7 cards)
    Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    hand1 = evaluate_hand(cards1)
    hand2 = evaluate_hand(cards2)

    if hand1 > hand2:
        return 1
    elif hand2 > hand1:
        return -1
    else:
        return 0


if __name__ == "__main__":
    # Test environment
    env = ShortDeckPokerEnv()
    state = env.reset()

    print("Initial state shape:", state.shape)
    print("Pot:", env.pot)
    print("Round:", env.round)
    print("Valid actions:", env.get_valid_actions())

    # Test hand evaluation
    hand = [0, 4, 8, 12, 16, 2, 6]  # Four Aces
    print("\nFour Aces:", evaluate_hand(hand))
