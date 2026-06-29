
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Env:
        def reset(self, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class gym:  # type: ignore
        Env = _Env

    class spaces:  # type: ignore
        Discrete = _Discrete
        Box = _Box


Card = Tuple[int, str]


class SimplifiedTexasHoldemEnv(gym.Env):
    """A small Gym-style poker environment for Phase 1."""

    CHECK_CALL = 0
    BET = 1
    FOLD = 2

    PREFLOP = "preflop"
    FLOP = "flop"

    def __init__(self):
        super().__init__()

        self.ranks = [10, 11, 12, 13, 14]
        self.suits = ["C", "D", "H", "S"]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1,
            high=20,
            shape=(9,),
            dtype=np.int32,
        )

        self.deck: List[Card] = []
        self.player_hands: List[List[Card]] = [[], []]
        self.community_cards: List[Card] = []
        self.pot = 2
        self.round = self.PREFLOP
        self.phase = self.PREFLOP
        self.current_player = 0
        self.has_active_bet = False
        self.bettor: Optional[int] = None
        self.done = False
        self.actions_this_round: List[Tuple[int, int]] = []

    def _create_deck(self) -> List[Card]:
        """Create a 20-card deck: 10, J, Q, K, A x 4 suits."""
        return [(rank, suit) for rank in self.ranks for suit in self.suits]

    def _shuffle_deck(self) -> None:
        """Shuffle using Gymnasium's seeded RNG when available."""
        if not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()
        self.np_random.shuffle(self.deck)

    def _deal_cards(self) -> None:
        """Deal 2 hole cards to each player."""
        self.player_hands = [[], []]
        self.player_hands[0] = [self.deck.pop(), self.deck.pop()]
        self.player_hands[1] = [self.deck.pop(), self.deck.pop()]

    def _card_to_id(self, card: Card) -> int:
        """Convert a card tuple into an integer ID from 0 to 19."""
        rank, suit = card
        return self.ranks.index(rank) * len(self.suits) + self.suits.index(suit)

    def _round_to_id(self) -> int:
        return 0 if self.round == self.PREFLOP else 1

    def _get_obs(self) -> np.ndarray:
        """Return Player 0 observation without exposing Player 1 hole cards."""
        hole_1 = self._card_to_id(self.player_hands[0][0])
        hole_2 = self._card_to_id(self.player_hands[0][1])

        flop_ids = [-1, -1, -1]
        for index, card in enumerate(self.community_cards[:3]):
            flop_ids[index] = self._card_to_id(card)

        return np.array(
            [
                hole_1,
                hole_2,
                flop_ids[0],
                flop_ids[1],
                flop_ids[2],
                self._round_to_id(),
                self.current_player,
                self.pot,
                int(self.has_active_bet),
            ],
            dtype=np.int32,
        )

    def reset(self, seed=None, options=None):
        """Reset the environment and return (state, info)."""
        super().reset(seed=seed)

        self.deck = self._create_deck()
        self._shuffle_deck()
        self._deal_cards()

        self.community_cards = []
        self.pot = 2
        self.round = self.PREFLOP
        self.phase = self.round
        self.current_player = 0
        self.has_active_bet = False
        self.bettor = None
        self.done = False
        self.actions_this_round = []

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Apply one action and return (next_state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")

        self._validate_action(action)

        reward = 0
        winner = None

        if self.has_active_bet:
            if action == self.CHECK_CALL:
                self.pot += 1
                self.actions_this_round.append((self.current_player, action))
                self.has_active_bet = False
                self.bettor = None
                reward, winner = self._finish_round_or_game()
            elif action == self.FOLD:
                winner = 1 - self.current_player
                reward = self._get_reward(winner)
                self.done = True
        else:
            if action == self.CHECK_CALL:
                self.actions_this_round.append((self.current_player, action))
                if self._both_players_checked():
                    reward, winner = self._finish_round_or_game()
                else:
                    self._switch_player()
            elif action == self.BET:
                self.pot += 1
                self.has_active_bet = True
                self.bettor = self.current_player
                self.actions_this_round.append((self.current_player, action))
                self._switch_player()

        info = self._get_info(winner=winner)
        return self._get_obs(), reward, self.done, info

    def _validate_action(self, action: int) -> None:
        if action not in [self.CHECK_CALL, self.BET, self.FOLD]:
            raise ValueError("Invalid action. Must be 0=Check/Call, 1=Bet, 2=Fold.")

        if self.has_active_bet:
            if action == self.BET:
                raise ValueError(
                    "Invalid action: cannot Bet when there is an active bet. "
                    "Use Call or Fold."
                )
        else:
            if action == self.FOLD:
                raise ValueError(
                    "Invalid action: cannot Fold when there is no active bet. "
                    "Use Check or Bet."
                )

    def _switch_player(self) -> None:
        self.current_player = 1 - self.current_player

    def _both_players_checked(self) -> bool:
        if len(self.actions_this_round) != 2:
            return False

        players = {player for player, _ in self.actions_this_round}
        actions = [action for _, action in self.actions_this_round]

        return players == {0, 1} and actions == [self.CHECK_CALL, self.CHECK_CALL]

    def _finish_round_or_game(self) -> Tuple[int, Optional[int]]:
        """End the current betting round. Reveal flop or showdown."""
        self.actions_this_round = []
        self.has_active_bet = False
        self.bettor = None

        if self.round == self.PREFLOP:
            self._reveal_flop()
            self.round = self.FLOP
            self.phase = self.round
            self.current_player = 0
            return 0, None

        winner = self._showdown()
        self.done = True

        return self._get_reward(winner), winner

    def _reveal_flop(self) -> None:
        if self.community_cards:
            return

        self.community_cards = [self.deck.pop() for _ in range(3)]

    def _showdown(self) -> Optional[int]:
        """Return 0 if Player 0 wins, 1 if Player 1 wins, or None for draw."""
        hand0 = self.player_hands[0] + self.community_cards
        hand1 = self.player_hands[1] + self.community_cards

        try:
            from hand_evaluator import compare_hands  # type: ignore

            result = compare_hands(hand0, hand1)

            if result in [0, 1, None]:
                return result

            if result == -1:
                return None
        except ImportError:
            pass

        return self._compare_hands(hand0, hand1)

    def _compare_hands(self, hand0: List[Card], hand1: List[Card]) -> Optional[int]:
        score0 = self._evaluate_hand(hand0)
        score1 = self._evaluate_hand(hand1)

        if score0 > score1:
            return 0

        if score1 > score0:
            return 1

        return None

    def _evaluate_hand(self, cards: List[Card]):
        """Evaluate exactly 5 cards and return a comparable tuple."""
        ranks = sorted([card[0] for card in cards], reverse=True)
        suits = [card[1] for card in cards]
        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)
        ranks_by_freq = sorted(
            rank_counts.keys(),
            key=lambda rank: (rank_counts[rank], rank),
            reverse=True,
        )

        is_flush = len(set(suits)) == 1
        unique_ranks = sorted(set(ranks))
        is_straight = (
            len(unique_ranks) == 5
            and unique_ranks[-1] - unique_ranks[0] == 4
        )
        straight_high = unique_ranks[-1] if is_straight else 0

        if unique_ranks == [10, 11, 12, 13, 14]:
            is_straight = True
            straight_high = 14

        if is_straight and is_flush:
            return (8, straight_high)

        if counts == [4, 1]:
            return (7, ranks_by_freq[0], ranks_by_freq[1])

        if counts == [3, 2]:
            return (6, ranks_by_freq[0], ranks_by_freq[1])

        if is_flush:
            return (5, *ranks)

        if is_straight:
            return (4, straight_high)

        if counts == [3, 1, 1]:
            return (3, ranks_by_freq[0], ranks_by_freq[1], ranks_by_freq[2])

        if counts == [2, 2, 1]:
            pairs = sorted(
                [rank for rank, count in rank_counts.items() if count == 2],
                reverse=True,
            )
            kicker = max(rank for rank, count in rank_counts.items() if count == 1)

            return (2, pairs[0], pairs[1], kicker)

        if counts == [2, 1, 1, 1]:
            pair = max(rank for rank, count in rank_counts.items() if count == 2)
            kickers = sorted(
                [rank for rank, count in rank_counts.items() if count == 1],
                reverse=True,
            )

            return (1, pair, *kickers)

        return (0, *ranks)

    def _get_reward(self, winner: Optional[int]) -> int:
        if winner == 0:
            return self.pot

        if winner == 1:
            return -self.pot

        return 0

    def _get_info(self, winner: Optional[int] = None) -> Dict[str, object]:
        return {
            "round": self.round,
            "phase": self.phase,
            "pot": self.pot,
            "current_player": self.current_player,
            "has_active_bet": self.has_active_bet,
            "bettor": self.bettor,
            "winner": winner,
        }


TexasHoldemEnv = SimplifiedTexasHoldemEnv
KuhnPokerEnv = SimplifiedTexasHoldemEnv