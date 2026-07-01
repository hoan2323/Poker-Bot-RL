import pytest
from hand_evaluator import (
    evaluate_five_card_hand,
    evaluate_best_hand,
    compare_hands,
    HIGH_CARD,
    ONE_PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    STRAIGHT_FLUSH,
)


def test_high_card():
    cards = [(14, "S"), (10, "H"), (8, "C"), (5, "D"), (2, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == HIGH_CARD
    assert res == (HIGH_CARD, 14, 10, 8, 5, 2)


def test_one_pair():
    cards = [(10, "S"), (10, "H"), (8, "C"), (5, "D"), (2, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == ONE_PAIR
    assert res == (ONE_PAIR, 10, 8, 5, 2)


def test_two_pair():
    cards = [(10, "S"), (10, "H"), (8, "C"), (8, "D"), (2, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == TWO_PAIR
    assert res == (TWO_PAIR, 10, 8, 2)


def test_three_of_a_kind():
    cards = [(10, "S"), (10, "H"), (10, "C"), (5, "D"), (2, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == THREE_OF_A_KIND
    assert res == (THREE_OF_A_KIND, 10, 5, 2)


def test_straight():
    cards = [(9, "S"), (8, "H"), (7, "C"), (6, "D"), (5, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == STRAIGHT
    assert res == (STRAIGHT, 9)


def test_ace_low_straight():
    cards = [(14, "S"), (2, "H"), (3, "C"), (4, "D"), (5, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == STRAIGHT
    assert res == (STRAIGHT, 5)


def test_flush():
    cards = [(14, "S"), (10, "S"), (8, "S"), (5, "S"), (2, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == FLUSH
    assert res == (FLUSH, 14, 10, 8, 5, 2)


def test_full_house():
    cards = [(10, "S"), (10, "H"), (10, "C"), (5, "D"), (5, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == FULL_HOUSE
    assert res == (FULL_HOUSE, 10, 5)


def test_four_of_a_kind():
    cards = [(10, "S"), (10, "H"), (10, "C"), (10, "D"), (2, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == FOUR_OF_A_KIND
    assert res == (FOUR_OF_A_KIND, 10, 2)


def test_straight_flush():
    cards = [(9, "S"), (8, "S"), (7, "S"), (6, "S"), (5, "S")]
    res = evaluate_five_card_hand(cards)
    assert res[0] == STRAIGHT_FLUSH
    assert res == (STRAIGHT_FLUSH, 9)


def test_compare_hands():
    p0 = [(14, "S"), (14, "H"), (10, "C"), (9, "D"), (8, "S"), (5, "C"), (2, "D")]  # Pair of Aces
    p1 = [(13, "S"), (13, "H"), (13, "C"), (9, "H"), (8, "H"), (5, "H"), (2, "H")]  # Three of a Kind Kings
    assert compare_hands(p0, p1) == 1

    p0_better = [(14, "S"), (14, "H"), (14, "C"), (14, "D"), (8, "S"), (5, "C"), (2, "D")]  # Four of a Kind Aces
    assert compare_hands(p0_better, p1) == 0

    # Draw: both players share community straight A-K-Q-J-10, hole cards can't improve
    p0_draw = [(2, "C"), (3, "D"), (14, "S"), (13, "D"), (12, "C"), (11, "H"), (10, "S")]
    p1_draw = [(4, "C"), (5, "D"), (14, "S"), (13, "D"), (12, "C"), (11, "H"), (10, "S")]
    assert compare_hands(p0_draw, p1_draw) == None


def test_evaluate_best_hand():
    # 7 cards: flush available
    cards = [(14, "S"), (10, "S"), (8, "S"), (5, "S"), (2, "S"), (2, "H"), (2, "D")]
    best = evaluate_best_hand(cards)
    assert best[0] == FLUSH