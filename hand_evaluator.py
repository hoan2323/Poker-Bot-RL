from collections import Counter

def _get_counts_and_ranks(cards):
    """Hàm phụ trợ để lấy danh sách rank và số lượng mỗi rank."""
    ranks = sorted([c[0] for c in cards], reverse=True)
    counts = Counter(ranks)
    return ranks, counts

def is_pair(cards):
    _, counts = _get_counts_and_ranks(cards)
    return list(counts.values()).count(2) == 1 and list(counts.values()).count(1) == 3

def is_two_pair(cards):
    _, counts = _get_counts_and_ranks(cards)
    return list(counts.values()).count(2) == 2

def is_three_of_kind(cards):
    _, counts = _get_counts_and_ranks(cards)
    return list(counts.values()).count(3) == 1 and list(counts.values()).count(1) == 2

def is_straight(cards):
    ranks, _ = _get_counts_and_ranks(cards)
    unique_ranks = sorted(list(set(ranks)))
    return len(unique_ranks) == 5 and unique_ranks[-1] - unique_ranks[0] == 4

def is_flush(cards):
    suits = set([c[1] for c in cards])
    return len(suits) == 1

def is_full_house(cards):
    _, counts = _get_counts_and_ranks(cards)
    return 3 in counts.values() and 2 in counts.values()

def is_four_of_kind(cards):
    _, counts = _get_counts_and_ranks(cards)
    return 4 in counts.values()

def is_straight_flush(cards):
    return is_straight(cards) and is_flush(cards)

def evaluate_hand(cards):
    """
    Trả về điểm số của hand dưới dạng tuple để dễ dàng so sánh.
    Tuple = (Loại hand, rank bài chính, rank kicker...)
    Số càng lớn càng mạnh.
    """
    ranks, counts = _get_counts_and_ranks(cards)
    # Sắp xếp rank theo số lượng xuất hiện, sau đó theo giá trị rank
    ranks_by_freq = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    if is_straight_flush(cards):
        return (8, ranks[0])
    elif is_four_of_kind(cards):
        return (7, ranks_by_freq[0], ranks_by_freq[1])
    elif is_full_house(cards):
        return (6, ranks_by_freq[0], ranks_by_freq[1])
    elif is_flush(cards):
        return (5, *ranks)
    elif is_straight(cards):
        return (4, ranks[0])
    elif is_three_of_kind(cards):
        return (3, ranks_by_freq[0], *sorted(ranks_by_freq[1:], reverse=True))
    elif is_two_pair(cards):
        pairs = sorted([ranks_by_freq[0], ranks_by_freq[1]], reverse=True)
        kicker = ranks_by_freq[2]
        return (2, pairs[0], pairs[1], kicker)
    elif is_pair(cards):
        return (1, ranks_by_freq[0], *sorted(ranks_by_freq[1:], reverse=True))
    else:
        return (0, *ranks)

def compare_hands(hand1, hand2):
    """
    So sánh 2 hand:
    Trả về 1 nếu hand1 thắng
    Trả về -1 nếu hand2 thắng
    Trả về 0 nếu hòa
    """
    score1 = evaluate_hand(hand1)
    score2 = evaluate_hand(hand2)

    if score1 > score2:
        return 1
    elif score1 < score2:
        return -1
    else:
        return 0