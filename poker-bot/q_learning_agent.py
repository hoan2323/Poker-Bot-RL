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
    def __init__(
        self,
        action_size,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        random_tie_break=False,
        fold_margin=0.25,
    ):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.random_tie_break = random_tie_break
        self.fold_margin = fold_margin
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

    def get_state_key(self, state, opponent_profile=0):
        hole_cards = [int(state[0]), int(state[1])]
        community_cards = [int(card) for card in state[2:7]]
        round_index = int(state[7])
        pot = int(state[9])
        has_active_bet = int(state[10])
        actions_this_round_count = int(state[11]) if len(state) > 11 else 0
        opponent_profile = int(opponent_profile)

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
            min(actions_this_round_count, 2),
            round_index,
            opponent_profile,
        )
        return state_key

    def _ensure_state(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float64)

    def choose_action(self, state, valid_actions, opponent_profile=0):
        if not valid_actions:
            raise ValueError("valid_actions must not be empty")

        state_key = self.get_state_key(state, opponent_profile)
        self._ensure_state(state_key)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_table[state_key]

        (
            made_hand_rank,
            best_current_hand_rank,
            _pair_flag,
            two_pair_flag,
            trips_flag,
            flush_draw,
            straight_draw,
            top_pair,
            overpair,
            _kicker_bucket,
            pot_bucket,
            has_active_bet,
            _actions_this_round_count,
            round_index,
            profile_bucket,
        ) = state_key

        strong_hand = (
            best_current_hand_rank >= 2
            or two_pair_flag
            or trips_flag
            or overpair
        )
        top_pair_only = bool(
            top_pair
            and made_hand_rank == ONE_PAIR
            and not two_pair_flag
            and not trips_flag
            and not overpair
        )
        medium_hand = (
            not strong_hand
            and (
                best_current_hand_rank >= 1
                or top_pair_only
                or made_hand_rank >= TWO_PAIR
                or flush_draw
                or straight_draw
            )
        )
        weak_hand = not strong_hand and not medium_hand
        call_station_profile = profile_bucket == 1
        aggressive_or_heuristic_profile = profile_bucket in (0, 2, 4)
        made_pair_or_better = made_hand_rank >= ONE_PAIR
        draw_only = (flush_draw or straight_draw) and not made_pair_or_better
        monster_hand = (
            best_current_hand_rank >= 2
            or trips_flag
            or (two_pair_flag and made_hand_rank >= TWO_PAIR)
        )

        if 0 in valid_actions and 2 in valid_actions:
            call_q = q_values[0]
            fold_q = q_values[2]

            if has_active_bet == 1:
                # 1) NEVER fold monster/strong hands
                if monster_hand:
                    return 0
                if strong_hand:
                    return 0
                # 2) Medium hands: almost never fold — only pure draws in very large pots
                if medium_hand and draw_only and pot_bucket >= 3 and fold_q >= call_q + 0.20:
                    return 2
                if medium_hand:
                    return 0
                # 3) Weak hands fold EASIER
                if weak_hand and pot_bucket >= 1 and fold_q >= call_q - 0.50:
                    return 2
                if weak_hand and round_index >= 1:
                    return 2

            if fold_q <= call_q + self.fold_margin:
                valid_actions = [action for action in valid_actions if action != 2]

        if 0 in valid_actions and 1 in valid_actions:
            if has_active_bet == 0:
                # 3) Force betting with strong/monster hands (wide threshold)
                if call_station_profile:
                    if monster_hand:
                        return 1
                    if strong_hand and q_values[1] >= q_values[0] - 0.60:
                        return 1
                    if made_pair_or_better and not draw_only and q_values[1] >= q_values[0] - 0.30:
                        return 1
                    if weak_hand or draw_only:
                        return 0

                # vs heuristic/aggressive/unknown — bet aggressively
                if monster_hand:
                    return 1
                if strong_hand and q_values[1] >= q_values[0] - 0.60:
                    return 1
                if medium_hand and q_values[1] >= q_values[0] - 0.30:
                    return 1
                if medium_hand and made_pair_or_better:
                    return 1
                # Weak hand bluffs — bet when Q suggests it's close
                if weak_hand and q_values[1] >= q_values[0] - 0.05:
                    return 1
                if weak_hand and round_index <= 0 and q_values[1] >= q_values[0] - 0.15:
                    return 1

        best_q = max(q_values[action] for action in valid_actions)
        best_actions = [action for action in valid_actions if q_values[action] == best_q]
        if self.random_tie_break:
            return random.choice(best_actions)
        return best_actions[0]

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        valid_next_actions,
        opponent_profile=0,
        next_opponent_profile=None,
    ):
        state_key = self.get_state_key(state, opponent_profile)
        self._ensure_state(state_key)

        current_q = self.q_table[state_key][action]

        if done or not valid_next_actions:
            target = reward
        else:
            if next_opponent_profile is None:
                next_opponent_profile = opponent_profile
            next_state_key = self.get_state_key(next_state, next_opponent_profile)
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