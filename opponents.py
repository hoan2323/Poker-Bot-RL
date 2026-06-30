import random

class RandomAgent:
    """Chọn action ngẫu nhiên trong các action hợp lệ."""
    def choose_action(self, state, valid_actions):
        return random.choice(valid_actions)

class FixedCheckAgent:
    """Luôn Check nếu có thể, nếu bị Bet thì Call."""
    def choose_action(self, state, valid_actions):
        # Action 0 đại diện cho cả Check và Call
        if 0 in valid_actions:
            return 0
        return valid_actions[0]

class FixedBetAgent:
    """Luôn Bet nếu có thể, nếu bị Bet thì Call."""
    def choose_action(self, state, valid_actions):
        if 1 in valid_actions:  # Có thể Bet
            return 1
        if 0 in valid_actions:  # Đã có active bet, thực hiện Call
            return 0
        return valid_actions[0]

class FixedFoldAgent:
    """Luôn Fold nếu có active bet, ngược lại thì Check."""
    def choose_action(self, state, valid_actions):
        if 2 in valid_actions:  # Có thể Fold (nghĩa là đang có active bet)
            return 2
        if 0 in valid_actions:  # Chưa có bet, thực hiện Check
            return 0
        return valid_actions[0]

class RuleBasedAgent:
    """
    Bot cơ bản dựa trên luật:
    - Bài mạnh: Bet hoặc Call.
    - Bài yếu: Check (nếu được) hoặc Fold (nếu bị bet).
    """
    def choose_action(self, state, valid_actions):
        # Trích xuất thông tin hole cards từ state
        hole_1_id = state[0]
        hole_2_id = state[1]
        
        # Chuyển ID thành Rank index (0-4 tương ứng 10-A)
        # Vì ID = rank_index * 4 + suit_index
        rank_1 = hole_1_id // 4
        rank_2 = hole_2_id // 4
        
        has_active_bet = state[8] == 1
        
        # Đánh giá bài tay: Mạnh nếu là đôi hoặc có lá cao (K, A tương ứng rank_index >= 3)
        is_strong = (rank_1 == rank_2) or (rank_1 >= 3) or (rank_2 >= 3)

        if has_active_bet:
            # Đang có bet -> valid_actions thường là [0(Call), 2(Fold)]
            if is_strong and 0 in valid_actions:
                return 0  # Call
            elif 2 in valid_actions:
                return 2  # Fold yếu
        else:
            # Không có bet -> valid_actions thường là [0(Check), 1(Bet)]
            if is_strong and 1 in valid_actions:
                return 1  # Bet mạnh
            elif 0 in valid_actions:
                return 0  # Check trung bình/yếu
        
        # Dự phòng
        return valid_actions[0]