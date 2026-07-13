from environment import (
    FOUR_OF_A_KIND,
    ONE_PAIR,
    STRAIGHT,
    compare_hands,
    evaluate_5_cards,
    evaluate_hand,
)


def test_evaluate_5_cards_recognizes_short_deck_straight():
    assert evaluate_5_cards([0, 5, 10, 15, 16])[0] == STRAIGHT


def test_evaluate_5_cards_recognizes_four_of_a_kind():
    assert evaluate_5_cards([0, 1, 2, 3, 4])[0] == FOUR_OF_A_KIND


def test_evaluate_hand_selects_best_five_cards():
    cards = [0, 1, 2, 3, 4, 8, 12]
    assert evaluate_hand(cards)[0] == FOUR_OF_A_KIND


def test_compare_hands_returns_one_zero_or_negative_one():
    four_of_a_kind = [0, 1, 2, 3, 4]
    one_pair = [4, 5, 8, 12, 17]
    tie = [0, 4, 8, 12, 16]

    assert evaluate_hand(one_pair)[0] == ONE_PAIR
    assert compare_hands(four_of_a_kind, one_pair) == 1
    assert compare_hands(one_pair, four_of_a_kind) == -1
    assert compare_hands(tie, tie) == 0
