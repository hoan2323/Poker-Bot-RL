import random
from collections import Counter
from itertools import combinations

import numpy as np

from hand_evaluator import (
    HIGH_CARD,
    ONE_PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    STRAIGHT_FLUSH,
    evaluate_best_hand,
    evaluate_five_card_hand,
)


_RANKS = [10, 11, 12, 13, 14]
_SUIT_MAP = {0: "C", 1: "D", 2: "H", 3: "S"}


class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _decode_card(self, card_id):
        if card_id == -1:
            return None, None
        rank_index = card_id // 4
        suit_index = card_id % 4
        rank = _RANKS[rank_index]
        return rank, _SUIT_MAP[suit_index]

    def _card_id_to_tuple(self, card_id):
        if card_id == -1:
            return None
        rank, suit = self._decode_card(card_id)
        return (rank, suit)

    def _known_cards(self, hole_cards, community_cards):
        return [
            self._card_id_to_tuple(card_id)
            for card_id in hole_cards + community_cards
            if card_id != -1
        ]

    def _ranks_and_suits(self, cards):
        ranks = [rank for rank, _ in cards]
        suits = [suit for _, suit in cards]
        return ranks, suits

    def _get_pot_bucket(self, pot):
        """0=2, 1=3-4, 2=5-6, 3=7+."""
        if pot <= 2:
            return 0
        if pot <= 4:
            return 1
        if pot <= 6:
            return 2
        return 3

    def _made_hand_rank(self, hole_cards, community_cards):
        """Best made hand rank using Player 0 visible cards: hole + revealed board."""
        known_cards = self._known_cards(hole_cards, community_cards)
        if len(known_cards) < 5:
            ranks, _ = self._ranks_and_suits(known_cards)
            if not ranks:
                return HIGH_CARD
            counts = Counter(ranks)
            pair_count = sum(1 for count in counts.values() if count >= 2)
            if any(count >= 3 for count in counts.values()):
                return THREE_OF_A_KIND
            if pair_count >= 2:
                return TWO_PAIR
            if pair_count == 1:
                return ONE_PAIR
            return HIGH_CARD

        if len(known_cards) == 7:
            return evaluate_best_hand(known_cards)[0]

        best = None
        for combo in combinations(known_cards, 5):
            score = evaluate_five_card_hand(list(combo))
            if best is None or score > best:
                best = score
        return best[0] if best is not None else HIGH_CARD

    def _best_current_hand_rank_bucket(self, made_hand_rank):
        """Compact rank bucket for current best hand."""
        if made_hand_rank <= ONE_PAIR:
            return 0
        if made_hand_rank == TWO_PAIR:
            return 1
        if made_hand_rank in (THREE_OF_A_KIND, STRAIGHT, FLUSH):
            return 2
        if made_hand_rank in (FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH):
            return 3
        return 0

    def _rank_counts(self, hole_cards, community_cards):
        known_cards = self._known_cards(hole_cards, community_cards)
        ranks, _ = self._ranks_and_suits(known_cards)
        return Counter(ranks)

    def _pair_flag(self, hole_cards, community_cards):
        counts = self._rank_counts(hole_cards, community_cards)
        return int(any(count >= 2 for count in counts.values()))

    def _two_pair_flag(self, hole_cards, community_cards):
        counts = self._rank_counts(hole_cards, community_cards)
        return int(sum(1 for count in counts.values() if count >= 2) >= 2)

    def _trips_flag(self, hole_cards, community_cards):
        counts = self._rank_counts(hole_cards, community_cards)
        return int(any(count >= 3 for count in counts.values()))

    def _has_flush_draw(self, hole_cards, community_cards):
        known_cards = self._known_cards(hole_cards, community_cards)
        _, suits = self._ranks_and_suits(known_cards)
        return int(any(suits.count(suit) >= 4 for suit in set(suits)))

    def _has_straight_draw(self, hole_cards, community_cards):
        known_cards = self._known_cards(hole_cards, community_cards)
        ranks, _ = self._ranks_and_suits(known_cards)
        if len(set(ranks)) < 4:
            return 0
        search_ranks = set(ranks)
        if 14 in search_ranks:
            search_ranks.add(1)
        ordered = sorted(search_ranks)
        for start in range(len(ordered) - 3):
            window = ordered[start : start + 4]
            if len(window) == 4 and window[-1] - window[0] <= 4:
                return 1
        return 0

    def _has_top_pair(self, hole_cards, community_cards):
        board_cards = [
            self._card_id_to_tuple(card_id)
            for card_id in community_cards
            if card_id != -1
        ]
        if not board_cards:
            return 0
        board_ranks = [rank for rank, _ in board_cards]
        top_board_rank = max(board_ranks)
        hole_ranks = [self._decode_card(card_id)[0] for card_id in hole_cards]
        return int(top_board_rank in hole_ranks)

    def _has_overpair(self, hole_cards, community_cards):
        board_cards = [
            self._card_id_to_tuple(card_id)
            for card_id in community_cards
            if card_id != -1
        ]
        if not board_cards:
            return 0
        hole_ranks = [self._decode_card(card_id)[0] for card_id in hole_cards]
        if len(hole_ranks) != 2 or hole_ranks[0] != hole_ranks[1]:
            return 0
        top_board_rank = max(rank for rank, _ in board_cards)
        return int(hole_ranks[0] > top_board_rank)

    def _kicker_bucket(self, hole_cards):
        hole_ranks = sorted(
            [self._decode_card(card_id)[0] for card_id in hole_cards],
            reverse=True,
        )
        kicker = hole_ranks[1]
        if kicker <= 5:
            return 0
        if kicker <= 9:
            return 1
        if kicker <= 12:
            return 2
        return 3

    def get_state_key(self, state):
        hole_cards = [int(state[0]), int(state[1])]
        community_cards = [int(card) for card in state[2:7]]
        round_index = int(state[7])
        pot = int(state[9])
        has_active_bet = int(state[10])

        made_hand_rank = self._made_hand_rank(hole_cards, community_cards)
        best_current_hand_rank = self._best_current_hand_rank_bucket(made_hand_rank)

        state_key = (
            made_hand_rank,
            best_current_hand_rank,
            self._pair_flag(hole_cards, community_cards),
            self._two_pair_flag(hole_cards, community_cards),
            self._trips_flag(hole_cards, community_cards),
            self._has_flush_draw(hole_cards, community_cards),
            self._has_straight_draw(hole_cards, community_cards),
            self._has_top_pair(hole_cards, community_cards),
            self._has_overpair(hole_cards, community_cards),
            self._kicker_bucket(hole_cards),
            self._get_pot_bucket(pot),
            has_active_bet,
            round_index,
        )
        return state_key

    def _ensure_state(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float64)

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("valid_actions must not be empty")

        state_key = self.get_state_key(state)
        self._ensure_state(state_key)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_table[state_key]
        return max(valid_actions, key=lambda action: q_values[action])

    def update(self, state, action, reward, next_state, done, valid_next_actions):
        state_key = self.get_state_key(state)
        self._ensure_state(state_key)

        current_q = self.q_table[state_key][action]

        if done or not valid_next_actions:
            target = reward
        else:
            next_state_key = self.get_state_key(next_state)
            self._ensure_state(next_state_key)
            next_q = max(
                self.q_table[next_state_key][next_action]
                for next_action in valid_next_actions
            )
            target = reward + self.gamma * next_q

        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)

    def save(self, path):
        np.save(path, self.q_table, allow_pickle=True)

    def load(self, path):
        self.q_table = np.load(path, allow_pickle=True).item()