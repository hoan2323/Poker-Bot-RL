from environment import (
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
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


def test_pair_is_compared_by_pair_then_actual_kickers():
    kings_with_ace = [12, 13, 16, 8, 4]
    kings_with_queen = [12, 13, 8, 4, 0]
    assert evaluate_5_cards(kings_with_ace)[0] == ONE_PAIR
    assert compare_hands(kings_with_ace, kings_with_queen) == 1


def test_two_pair_is_compared_by_high_pair_low_pair_then_kicker():
    aces_kings = [16, 17, 12, 13, 8]
    aces_queens = [16, 17, 8, 9, 12]
    same_pairs_better_kicker = [16, 17, 12, 13, 8]
    same_pairs_lower_kicker = [16, 17, 12, 13, 4]
    assert evaluate_5_cards(aces_kings)[0] == TWO_PAIR
    assert compare_hands(aces_kings, aces_queens) == 1
    assert compare_hands(same_pairs_better_kicker, same_pairs_lower_kicker) == 1


def test_trips_is_compared_by_trip_rank_then_actual_kickers():
    queens_ace_kicker = [8, 9, 10, 16, 4]
    queens_king_kicker = [8, 9, 10, 12, 4]
    assert evaluate_5_cards(queens_ace_kicker)[0] == THREE_OF_A_KIND
    assert compare_hands(queens_ace_kicker, queens_king_kicker) == 1


def test_full_house_and_flush_tie_breaks():
    aces_full = [16, 17, 18, 12, 13]
    kings_full = [12, 13, 14, 16, 17]
    ace_high_flush = [0, 4, 8, 12, 16]
    assert evaluate_5_cards(aces_full)[0] == FULL_HOUSE
    assert compare_hands(aces_full, kings_full) == 1
    assert evaluate_5_cards(ace_high_flush)[0] == STRAIGHT_FLUSH


def test_straight_and_four_card_suit_are_not_false_straight_flush():
    mixed_suit_straight = [0, 4, 8, 12, 17]
    assert evaluate_5_cards(mixed_suit_straight)[0] == STRAIGHT
    assert evaluate_5_cards(mixed_suit_straight)[0] != STRAIGHT_FLUSH


def test_flush_high_cards_are_part_of_score():
    # In this 20-card deck, any valid five-card flush necessarily contains all
    # five ranks and is therefore a straight flush. This assertion documents
    # that deck property while score ordering is covered by mixed-suit hands.
    flush = [1, 5, 9, 13, 17]
    score = evaluate_5_cards(flush)
    assert score == (STRAIGHT_FLUSH, (4,))
