import sys
import traceback

from texas_holdenv import KuhnPokerEnv, SimplifiedTexasHoldemEnv, TexasHoldemEnv

PASS = "PASS"
FAIL = "FAIL"
results = []


def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print("-" * 60)
    try:
        fn()
        print(f">>> {PASS}")
        results.append((name, PASS, None))
    except Exception as e:
        print(f">>> {FAIL}: {e}")
        traceback.print_exc()
        results.append((name, FAIL, str(e)))


def print_state(label, env, state=None, reward=None, done=None, info=None):
    print(f"[{label}]")
    print("  player_hands:", env.player_hands)
    print("  community_cards:", env.community_cards)
    print("  pot:", env.pot)
    print("  round:", env.round)
    print("  phase:", env.phase)
    print("  current_player:", env.current_player)
    print("  has_active_bet:", env.has_active_bet)
    print("  bettor:", env.bettor)
    print("  done:", env.done)
    if state is not None:
        print("  state:", state.tolist())
    if reward is not None:
        print("  reward:", reward)
    if done is not None:
        print("  done returned:", done)
    if info is not None:
        print("  info:", info)


# ── tests ────────────────────────────────────────────────────────────────────

def test_class_aliases_import():
    assert TexasHoldemEnv is SimplifiedTexasHoldemEnv, "TexasHoldemEnv alias wrong"
    assert KuhnPokerEnv is SimplifiedTexasHoldemEnv, "KuhnPokerEnv alias wrong"
    print("  TexasHoldemEnv is SimplifiedTexasHoldemEnv:", TexasHoldemEnv is SimplifiedTexasHoldemEnv)
    print("  KuhnPokerEnv  is SimplifiedTexasHoldemEnv:", KuhnPokerEnv is SimplifiedTexasHoldemEnv)


def test_reset_runs():
    env = SimplifiedTexasHoldemEnv()
    state, info = env.reset()
    print_state("after reset", env, state=state, info=info)
    assert state is not None
    assert state.shape == (9,), f"shape={state.shape}"
    assert isinstance(info, dict)


def test_action_space():
    env = SimplifiedTexasHoldemEnv()
    print("  action_space.n:", env.action_space.n)
    assert env.action_space.n == 3


def test_deck_has_20_unique_cards():
    env = SimplifiedTexasHoldemEnv()
    deck = env._create_deck()
    print("  deck:", deck)
    print("  len:", len(deck), "| unique:", len(set(deck)))
    assert len(deck) == 20
    assert len(deck) == len(set(deck))


def test_reset_deals_two_cards_each_player_and_no_flop():
    env = SimplifiedTexasHoldemEnv()
    state, info = env.reset()
    print_state("after reset", env, state=state, info=info)
    assert len(env.player_hands[0]) == 2
    assert len(env.player_hands[1]) == 2
    assert env.community_cards == []
    assert state[2] == -1
    assert state[3] == -1
    assert state[4] == -1


def test_no_duplicate_cards_after_deal():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    cards = env.player_hands[0] + env.player_hands[1]
    print("  dealt cards:", cards)
    print("  total:", len(cards), "| unique:", len(set(cards)))
    assert len(cards) == len(set(cards))


def test_reset_initial_state_values():
    env = SimplifiedTexasHoldemEnv()
    state, info = env.reset()
    print_state("after reset", env, state=state, info=info)
    assert env.pot == 2
    assert env.round == "preflop"
    assert env.phase == "preflop"
    assert env.current_player == 0
    assert env.has_active_bet is False
    assert env.bettor is None
    assert env.done is False
    assert env.actions_this_round == []
    assert state[5] == 0
    assert state[6] == 0
    assert state[7] == 2
    assert state[8] == 0


def test_fold_invalid_without_active_bet():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    print_state("before invalid fold", env)
    try:
        env.step(2)
        raise AssertionError("Expected ValueError not raised")
    except ValueError as e:
        print("  Caught expected ValueError:", e)


def test_bet_invalid_when_active_bet_exists():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    state, reward, done, info = env.step(1)
    print_state("after P0 bet", env, state=state, reward=reward, done=done, info=info)
    try:
        env.step(1)
        raise AssertionError("Expected ValueError not raised")
    except ValueError as e:
        print("  Caught expected ValueError:", e)


def test_bet_creates_active_bet_and_increases_pot():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    old_pot = env.pot
    state, reward, done, info = env.step(1)
    print_state("after P0 bet", env, state=state, reward=reward, done=done, info=info)
    print("  old_pot:", old_pot, "-> new_pot:", env.pot)
    assert env.pot == old_pot + 1
    assert env.has_active_bet is True
    assert env.bettor == 0
    assert env.current_player == 1
    assert reward == 0
    assert done is False


def test_check_check_preflop_reveals_flop():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    state, reward, done, info = env.step(0)
    print_state("P0 check", env, state=state, reward=reward, done=done, info=info)
    state, reward, done, info = env.step(0)
    print_state("P1 check -> flop", env, state=state, reward=reward, done=done, info=info)
    assert env.round == "flop"
    assert env.phase == "flop"
    assert len(env.community_cards) == 3
    assert env.current_player == 0
    assert env.has_active_bet is False
    assert reward == 0
    assert done is False
    assert state[2] != -1
    assert state[3] != -1
    assert state[4] != -1


def test_bet_call_preflop_reveals_flop():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    state, reward, done, info = env.step(1)
    print_state("P0 bet", env, state=state, reward=reward, done=done, info=info)
    state, reward, done, info = env.step(0)
    print_state("P1 call -> flop", env, state=state, reward=reward, done=done, info=info)
    assert env.round == "flop"
    assert len(env.community_cards) == 3
    assert env.pot == 4
    assert env.has_active_bet is False
    assert reward == 0
    assert done is False


def test_fold_after_bet_ends_game_immediately():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    env.step(1)
    state, reward, done, info = env.step(2)
    print_state("P1 fold -> game over", env, state=state, reward=reward, done=done, info=info)
    assert done is True
    assert env.done is True
    assert info["winner"] == 0
    assert reward == env.pot


def test_player0_fold_after_player1_bet_loses():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    env.step(0)
    env.step(1)
    state, reward, done, info = env.step(2)
    print_state("P0 fold -> P1 wins", env, state=state, reward=reward, done=done, info=info)
    assert done is True
    assert info["winner"] == 1
    assert reward == -env.pot


def test_check_check_flop_triggers_showdown():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    env.step(0)
    env.step(0)
    state, reward, done, info = env.step(0)
    print_state("P0 check flop", env, state=state, reward=reward, done=done, info=info)
    state, reward, done, info = env.step(0)
    print_state("P1 check flop -> showdown", env, state=state, reward=reward, done=done, info=info)
    assert done is True
    assert env.done is True
    assert reward in [-env.pot, 0, env.pot]
    assert info["winner"] in [0, 1, None]


def test_bet_call_flop_triggers_showdown():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    env.step(0)
    env.step(0)
    state, reward, done, info = env.step(1)
    print_state("P0 bet flop", env, state=state, reward=reward, done=done, info=info)
    state, reward, done, info = env.step(0)
    print_state("P1 call flop -> showdown", env, state=state, reward=reward, done=done, info=info)
    assert done is True
    assert env.done is True
    assert env.pot == 4
    assert reward in [-env.pot, 0, env.pot]
    assert info["winner"] in [0, 1, None]


def test_invalid_action_number_raises_error():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    print_state("before invalid action 99", env)
    try:
        env.step(99)
        raise AssertionError("Expected ValueError not raised")
    except ValueError as e:
        print("  Caught expected ValueError:", e)


def test_step_after_done_raises_error():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    env.step(1)
    env.step(2)
    print_state("game done", env)
    try:
        env.step(0)
        raise AssertionError("Expected RuntimeError not raised")
    except RuntimeError as e:
        print("  Caught expected RuntimeError:", e)


def test_reset_seed_is_reproducible():
    env1 = SimplifiedTexasHoldemEnv()
    env2 = SimplifiedTexasHoldemEnv()
    state1, _ = env1.reset(seed=123)
    state2, _ = env2.reset(seed=123)
    print("  state1:", state1.tolist())
    print("  state2:", state2.tolist())
    print("  env1 player_hands:", env1.player_hands)
    print("  env2 player_hands:", env2.player_hands)
    assert state1.tolist() == state2.tolist()
    assert env1.player_hands == env2.player_hands


def test_no_turn_or_river_exists_after_flop_showdown():
    env = SimplifiedTexasHoldemEnv()
    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    state, reward, done, info = env.step(0)
    print_state("flop showdown", env, state=state, reward=reward, done=done, info=info)
    assert done is True
    assert env.round == "flop"
    assert len(env.community_cards) == 3


# ── run all ──────────────────────────────────────────────────────────────────

TESTS = [
    ("class_aliases_import",                    test_class_aliases_import),
    ("reset_runs",                              test_reset_runs),
    ("action_space",                            test_action_space),
    ("deck_has_20_unique_cards",                test_deck_has_20_unique_cards),
    ("reset_deals_two_cards_each_player",       test_reset_deals_two_cards_each_player_and_no_flop),
    ("no_duplicate_cards_after_deal",           test_no_duplicate_cards_after_deal),
    ("reset_initial_state_values",              test_reset_initial_state_values),
    ("fold_invalid_without_active_bet",         test_fold_invalid_without_active_bet),
    ("bet_invalid_when_active_bet_exists",      test_bet_invalid_when_active_bet_exists),
    ("bet_creates_active_bet_increases_pot",    test_bet_creates_active_bet_and_increases_pot),
    ("check_check_preflop_reveals_flop",        test_check_check_preflop_reveals_flop),
    ("bet_call_preflop_reveals_flop",           test_bet_call_preflop_reveals_flop),
    ("fold_after_bet_ends_game",                test_fold_after_bet_ends_game_immediately),
    ("player0_fold_after_player1_bet",          test_player0_fold_after_player1_bet_loses),
    ("check_check_flop_triggers_showdown",      test_check_check_flop_triggers_showdown),
    ("bet_call_flop_triggers_showdown",         test_bet_call_flop_triggers_showdown),
    ("invalid_action_raises_error",             test_invalid_action_number_raises_error),
    ("step_after_done_raises_error",            test_step_after_done_raises_error),
    ("reset_seed_reproducible",                 test_reset_seed_is_reproducible),
    ("no_turn_river_after_flop_showdown",       test_no_turn_or_river_exists_after_flop_showdown),
]

if __name__ == "__main__":
    for name, fn in TESTS:
        run_test(name, fn)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r, _ in results if r == PASS)
    failed = sum(1 for _, r, _ in results if r == FAIL)
    for name, result, err in results:
        mark = "✓" if result == PASS else "✗"
        line = f"  {mark} {name}"
        if err:
            line += f"  →  {err}"
        print(line)
    print(f"\n  Passed: {passed}/{len(results)}  |  Failed: {failed}/{len(results)}")
    sys.exit(0 if failed == 0 else 1)