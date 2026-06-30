from opponents import RandomAgent, FixedCheckAgent, FixedBetAgent, FixedFoldAgent, RuleBasedAgent

def test_opponents():
    # Giả lập state array có 9 phần tử
    # [hole_1, hole_2, flop_1, flop_2, flop_3, round, current_player, pot, has_active_bet]
    
    # ID bài: rank_index * 4 + suit_index. Rank: 0=10, 1=J, 2=Q, 3=K, 4=A
    # Bài mạnh: 2 lá A (ID 18, 19)
    state_strong_no_bet = [19, 18, -1, -1, -1, 0, 0, 2, 0] 
    state_strong_with_bet = [19, 18, -1, -1, -1, 0, 0, 3, 1] 
    
    # Bài yếu: 10 bích (ID 3), J tép (ID 4)
    state_weak_no_bet = [3, 4, -1, -1, -1, 0, 0, 2, 0] 
    state_weak_with_bet = [3, 4, -1, -1, -1, 0, 0, 3, 1]

    # Các action hợp lệ theo rules
    actions_no_bet = [0, 1] # Check, Bet
    actions_with_bet = [0, 2] # Call, Fold

    print("Đang test FixedCheckAgent...")
    check_agent = FixedCheckAgent()
    assert check_agent.choose_action(state_strong_no_bet, actions_no_bet) == 0, "Có thể Check thì phải Check"
    assert check_agent.choose_action(state_strong_with_bet, actions_with_bet) == 0, "Bị Bet thì phải Call"

    print("Đang test FixedBetAgent...")
    bet_agent = FixedBetAgent()
    assert bet_agent.choose_action(state_strong_no_bet, actions_no_bet) == 1, "Có thể Bet thì phải Bet"
    assert bet_agent.choose_action(state_strong_with_bet, actions_with_bet) == 0, "Không Bet được thì phải Call"

    print("Đang test FixedFoldAgent...")
    fold_agent = FixedFoldAgent()
    assert fold_agent.choose_action(state_strong_with_bet, actions_with_bet) == 2, "Có active bet thì phải Fold"
    assert fold_agent.choose_action(state_strong_no_bet, actions_no_bet) == 0, "Không bị bet thì phải Check"

    print("Đang test RuleBasedAgent...")
    rule_agent = RuleBasedAgent()
    
    # Test bài mạnh
    assert rule_agent.choose_action(state_strong_no_bet, actions_no_bet) == 1, "Bài mạnh, không bị bet -> phải Bet"
    assert rule_agent.choose_action(state_strong_with_bet, actions_with_bet) == 0, "Bài mạnh, bị bet -> phải Call"
    
    # Test bài yếu
    assert rule_agent.choose_action(state_weak_no_bet, actions_no_bet) == 0, "Bài yếu, không bị bet -> phải Check"
    assert rule_agent.choose_action(state_weak_with_bet, actions_with_bet) == 2, "Bài yếu, bị bet -> phải Fold"

    print(">>> PASS: Tất cả hành vi bot opponents đều đúng luật!")

if __name__ == "__main__":
    test_opponents()