from hand_evaluator import evaluate_hand, compare_hands

def test_hand_evaluator():
    # Định nghĩa các hand bài (5 lá: 2 tay + 3 chung)
    
    # 1. Straight Flush: 10, J, Q, K, A bích
    straight_flush = [(10, 'S'), (11, 'S'), (12, 'S'), (13, 'S'), (14, 'S')]
    
    # 2. Tứ quý Át (Four of a Kind) + 1 kicker
    four_aces = [(14, 'S'), (14, 'H'), (14, 'D'), (14, 'C'), (13, 'S')]
    
    # 3. Cù lũ (Full House): 3 con K, 2 con Q
    full_house = [(13, 'S'), (13, 'H'), (13, 'D'), (12, 'C'), (12, 'S')]
    
    # 4. Thùng (Flush): 5 lá bích không liên tiếp
    flush = [(14, 'S'), (12, 'S'), (11, 'S'), (10, 'S'), (10, 'C')] 
    flush[4] = (13, 'H') # Fix lại để không thành Straight Flush
    flush_real = [(14, 'S'), (12, 'S'), (11, 'S'), (10, 'S'), (13, 'S')] # Sửa lại cho đúng flush
    flush_real = [(14, 'C'), (12, 'C'), (11, 'C'), (10, 'C'), (10, 'D')] # Sửa lại lần nữa
    flush_valid = [(14, 'C'), (12, 'C'), (11, 'C'), (10, 'C'), (13, 'C')] # Đây là sảnh thùng
    # Sửa lại Flush chuẩn: A, Q, J, 10, 10 không được, 5 lá phải cùng chất.
    # Với deck 20 lá, 5 lá cùng chất chắc chắn là sảnh thùng (Straight Flush). 
    # Môi trường 20 lá không thể có Flush bình thường mà không phải Straight Flush.
    # Do đó ta bỏ qua test Flush thường.

    # 5. Sảnh (Straight): 10-A khác chất
    straight = [(10, 'S'), (11, 'H'), (12, 'D'), (13, 'C'), (14, 'S')]
    
    # 6. Sám cô (Three of a Kind): 3 con Q
    three_of_kind = [(12, 'S'), (12, 'H'), (12, 'D'), (14, 'C'), (11, 'S')]
    
    # 7. Thú (Two Pair): 2 đôi K và J
    two_pair = [(13, 'S'), (13, 'H'), (11, 'D'), (11, 'C'), (14, 'S')]
    
    # 8. Một đôi (Pair): Đôi 10
    one_pair = [(10, 'S'), (10, 'H'), (14, 'D'), (13, 'C'), (12, 'S')]
    
    # 9. Mậu thầu (High Card) - Trong deck 20 lá, bốc 5 lá kiểu gì cũng có ít nhất sảnh hoặc đôi,
    # rất khó tạo High Card thuần, ta sẽ tập trung test các hand trên.

    print("Đang test so sánh các loại hand...")
    assert compare_hands(straight_flush, four_aces) == 1, "Straight Flush phải thắng Tứ quý"
    assert compare_hands(four_aces, full_house) == 1, "Tứ quý phải thắng Cù lũ"
    assert compare_hands(full_house, straight) == 1, "Cù lũ phải thắng Sảnh"
    assert compare_hands(straight, three_of_kind) == 1, "Sảnh phải thắng Sám cô"
    assert compare_hands(three_of_kind, two_pair) == 1, "Sám cô phải thắng 2 Đôi"
    assert compare_hands(two_pair, one_pair) == 1, "2 Đôi phải thắng 1 Đôi"
    
    print("Đang test Tie-break (Hòa loại bài, so Kicker)...")
    # Cùng 2 đôi, so kicker
    two_pair_kicker_A = [(13, 'S'), (13, 'H'), (11, 'D'), (11, 'C'), (14, 'S')] # Kicker A
    two_pair_kicker_Q = [(13, 'C'), (13, 'D'), (11, 'S'), (11, 'H'), (12, 'S')] # Kicker Q
    assert compare_hands(two_pair_kicker_A, two_pair_kicker_Q) == 1, "Kicker Át phải thắng Kicker Q"
    
    # Cùng đôi, so đôi cao hơn
    pair_K = [(13, 'S'), (13, 'H'), (14, 'D'), (12, 'C'), (11, 'S')]
    pair_Q = [(12, 'S'), (12, 'H'), (14, 'D'), (13, 'C'), (11, 'S')]
    assert compare_hands(pair_K, pair_Q) == 1, "Đôi K phải thắng đôi Q"
    
    # Hòa 100%
    pair_K_draw = [(13, 'C'), (13, 'D'), (14, 'C'), (12, 'H'), (11, 'D')]
    assert compare_hands(pair_K, pair_K_draw) == 0, "Hai bài y hệt rank phải hòa"

    print(">>> PASS: Tất cả logic của hand_evaluator đều đúng!")

if __name__ == "__main__":
    test_hand_evaluator()