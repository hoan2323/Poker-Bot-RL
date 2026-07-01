"""
hand_evaluator.py — Poker hand evaluation for Texas Hold'em.
Supports all standard hand rankings including ace-low straight.
"""

from itertools import combinations
from collections import Counter


# Hand rank constants
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


def evaluate_five_card_hand(cards):
    """
    Evaluate a 5-card poker hand.

    Input:
        cards: list of exactly 5 cards [(rank, suit), ...]

    Output:
        comparable tuple — higher is better
    """
    assert len(cards) == 5, "Must provide exactly 5 cards"

    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    unique_ranks = sorted(rank_counts.keys(), reverse=True)

    is_flush = len(set(suits)) == 1

    # Check straight
    is_straight = False
    straight_high = None

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Ace-low straight: A, 2, 3, 4, 5 (ranks: 14, 2, 3, 4, 5)
        elif set(ranks) == {14, 2, 3, 4, 5}:
            is_straight = True
            straight_high = 5  # 5-high straight

    # Straight Flush
    if is_straight and is_flush:
        return (STRAIGHT_FLUSH, straight_high)

    # Four of a Kind
    if counts[0] == 4:
        quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return (FOUR_OF_A_KIND, quad_rank, kicker)

    # Full House
    if counts[0] == 3 and counts[1] == 2:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        return (FULL_HOUSE, trip_rank, pair_rank)

    # Flush
    if is_flush:
        return (FLUSH,) + tuple(ranks)

    # Straight
    if is_straight:
        return (STRAIGHT, straight_high)

    # Three of a Kind
    if counts[0] == 3:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (THREE_OF_A_KIND, trip_rank) + tuple(kickers)

    # Two Pair
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return (TWO_PAIR, pairs[0], pairs[1], kicker)

    # One Pair
    if counts[0] == 2:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (ONE_PAIR, pair_rank) + tuple(kickers)

    # High Card
    return (HIGH_CARD,) + tuple(ranks)


def evaluate_best_hand(cards):
    """
    Find the best 5-card hand from 7 cards.

    Input:
        cards: list of 7 cards [(rank, suit), ...]

    Output:
        best comparable hand score among all 5-card combinations
    """
    assert len(cards) == 7, "Must provide exactly 7 cards"
    best = None
    for combo in combinations(cards, 5):
        score = evaluate_five_card_hand(list(combo))
        if best is None or score > best:
            best = score
    return best


def compare_hands(player0_cards, player1_cards):
    """
    Compare two 7-card hands.

    Input:
        player0_cards: list of 7 cards
        player1_cards: list of 7 cards

    Return:
        0 if Player 0 wins
        1 if Player 1 wins
        None if draw
    """
    score0 = evaluate_best_hand(player0_cards)
    score1 = evaluate_best_hand(player1_cards)

    if score0 > score1:
        return 0
    elif score1 > score0:
        return 1
    else:
        return None