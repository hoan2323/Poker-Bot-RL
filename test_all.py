"""
test_all.py - Test all components
"""

import sys


def test_environment():
    """Test environment"""
    print("\n" + "="*50)
    print("Testing Environment...")
    print("="*50)

    from environment import ShortDeckPokerEnv, evaluate_hand, compare_hands

    env = ShortDeckPokerEnv()
    state = env.reset()

    print(f"State shape: {state.shape}")
    print(f"Valid actions: {env.get_valid_actions()}")
    print(f"Pot: {env.pot}")

    # Test hand evaluation
    print("\n--- Hand Evaluation ---")
    test_cases = [
        [0, 4, 8, 12, 16, 2, 6],  # Four Aces
        [0, 1, 2, 3, 4, 10, 15],   # Straight
        [0, 1, 2, 3, 4, 5, 6],       # Flush
    ]

    for cards in test_cases:
        result = evaluate_hand(cards)
        print(f"Hand: {result}")

    print("\n✓ Environment OK")
    return True


def test_networks():
    """Test neural networks"""
    print("\n" + "="*50)
    print("Testing Networks...")
    print("="*50)

    import torch
    from networks import QNetwork, AveragePolicyNetwork

    q_net = QNetwork()
    policy_net = AveragePolicyNetwork()

    state = torch.randn(186)

    q_values = q_net(state)
    probs = policy_net(state)

    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    print(f"Probabilities sum: {probs.sum():.3f}")

    print("\n✓ Networks OK")
    return True


def test_memory():
    """Test memory components"""
    print("\n" + "="*50)
    print("Testing Memory...")
    print("="*50)

    import numpy as np
    from replay_buffer import ReplayBuffer
    from reservoir import ReservoirSampling

    # Test replay buffer
    buffer = ReplayBuffer(capacity=100)

    for i in range(50):
        state = np.random.randn(186)
        buffer.add(state, 0, 0.0, state, False)

    print(f"Replay buffer size: {len(buffer)}")
    batch = buffer.sample(8)
    print(f"Sampled batch: {len(batch)}")

    # Test reservoir
    reservoir = ReservoirSampling(capacity=100)

    for i in range(200):
        state = np.random.randn(186)
        policy = np.random.dirichlet([1, 1, 1])
        reservoir.add(state, policy)

    print(f"Reservoir size: {len(reservoir)}")
    batch = reservoir.sample(8)
    print(f"Sampled batch: {len(batch)}")

    print("\n✓ Memory OK")
    return True


def test_agent():
    """Test NFSP agent"""
    print("\n" + "="*50)
    print("Testing NFSP Agent...")
    print("="*50)

    from nfsp_agent import NSFPAgent
    import numpy as np

    agent = NSFPAgent()

    # Test choose action
    state = np.random.randn(186)
    action = agent.choose_action(state)
    print(f"Chosen action: {action}")

    # Store experiences
    for _ in range(100):
        state = np.random.randn(186)
        agent.store_experience(state, 0, 0.0, state, False)

    print(f"RL buffer: {len(agent.rl_buffer)}")
    print(f"SL reservoir: {len(agent.sl_reservoir)}")

    print("\n✓ Agent OK")
    return True


def test_evaluation():
    """Test evaluation"""
    print("\n" + "="*50)
    print("Testing Evaluation...")
    print("="*50)

    from evaluate import evaluate_agent, OPPONENTS
    from nfsp_agent import NSFPAgent

    agent = NSFPAgent()

    # Test against random (quick)
    wr = evaluate_agent(agent, 'random', n_games=50, verbose=False)
    print(f"vs Random win rate: {wr:.2%}")

    print("\n✓ Evaluation OK")
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("NFSP POKER BOT - COMPONENT TESTS")
    print("="*50)

    tests = [
        ("Environment", test_environment),
        ("Networks", test_networks),
        ("Memory", test_memory),
        ("Agent", test_agent),
        ("Evaluation", test_evaluation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(r for _, r in results)

    print("\n" + "="*50)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*50)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
