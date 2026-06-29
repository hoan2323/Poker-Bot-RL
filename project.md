# PROJECT_SPEC.md

# Poker Bot RL — Phase 1: Texas Hold'em Environment

## Goal

Implement only the game environment for a simplified Texas Hold'em Poker bot.

This phase does **not** include:

* Q-Learning
* Training
* Evaluation
* Q-table
* Visualization

The only required files for this phase are:

```text
texas_holdenv.py
test_env.py
```

The final result must pass:

```bash
pytest test_env.py
```

---

## Game Rules

Use a standard 52-card deck.

```python
ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
suits = ["C", "D", "H", "S"]
```

Rank mapping:

```text
11 = J
12 = Q
13 = K
14 = A
```

Card format:

```python
(rank, suit)
```

Example:

```python
(14, "S")  # Ace of Spades
(10, "H")  # Ten of Hearts
```

There are 2 players:

```text
Player 0: main agent
Player 1: opponent
```

Reward is always from Player 0's perspective:

```text
Player 0 wins  -> +2
Player 0 loses -> -2
Draw           -> 0
```

---

## Actions

The environment has 3 actions:

```text
0 = Check
1 = Bet
2 = Fold
```

Declare:

```python
self.action_space = spaces.Discrete(3)
```

Action behavior:

| Action | Meaning                          |
| ------ | -------------------------------- |
| Check  | Pass turn without betting        |
| Bet    | Add 1 chip to the pot            |
| Fold   | Current player loses immediately |

---

## Observation

Player 0 must not see Player 1's cards.

Observation format:

```text
[hole_1, hole_2, community_1, community_2, community_3, phase, current_player]
```

Use `-1` for unknown community cards.

Phase values:

```text
0 = preflop
1 = flop
2 = showdown
```

Current player values:

```text
0 = Player 0
1 = Player 1
```

Observation space:

```python
self.observation_space = spaces.Box(
    low=-1,
    high=51,
    shape=(7,),
    dtype=np.int32
)
```

Cards should be encoded as integers from `0` to `51`.

---

## File: `texas_holdenv.py`

Create a Gymnasium-compatible environment:

```python
class TexasHoldemEnv(gym.Env):
```

Optional alias for compatibility:

```python
KuhnPokerEnv = TexasHoldemEnv
```

Required methods:

```python
__init__()
reset()
step(action)
_create_deck()
_card_to_id(card)
_get_obs()
_reveal_flop()
_resolve_showdown()
```

Internal state:

```python
self.deck
self.player_hands
self.community_cards
self.pot
self.phase
self.current_player
self.done
self.round_actions
```

---

## Task 1 — Initialize Environment and `reset()`

Implement:

```python
__init__()
_create_deck()
_card_to_id(card)
_get_obs()
reset()
```

`reset()` must:

1. Create a 52-card deck
2. Shuffle the deck
3. Deal 2 cards to Player 0
4. Deal 2 cards to Player 1
5. Set community cards to empty
6. Set pot to `2`
7. Set phase to `"preflop"`
8. Set current player to `0`
9. Set done to `False`
10. Return `(state, info)`

Task 1 is complete when:

```text
[ ] texas_holdenv.py exists
[ ] TexasHoldemEnv can be imported
[ ] action_space is Discrete(3)
[ ] observation_space is valid
[ ] reset() runs without error
[ ] each player receives 2 cards
[ ] reset() returns state and info
```

---

## Task 2 — Implement `step()` and Flop Logic

`step(action)` must return:

```python
state, reward, terminated, truncated, info
```

`truncated` should always be `False`.

### Fold

If the current player folds:

```text
The other player wins.
The game ends immediately.
```

Reward:

```text
Player 0 wins  -> +2
Player 0 loses -> -2
```

### Check / Bet

If action is Check:

```text
Do not change pot.
Save action.
Move to next player or next phase.
```

If action is Bet:

```text
pot += 1
Save action.
Move to next player or next phase.
```

### Preflop

Both players act once.

If nobody folds:

```text
Reveal 3 community cards.
phase = "flop"
current_player = 0
round_actions = []
```

### Flop

Both players act once.

If nobody folds:

```text
Go to showdown.
Compare hands.
End the game.
```

### Showdown

Each player has 5 cards:

```text
2 hole cards + 3 community cards
```

Determine the winner.

For this phase, a simple hand evaluator can be implemented directly inside `texas_holdenv.py`.

Task 2 is complete when:

```text
[ ] step(0) works
[ ] step(1) works
[ ] step(2) works
[ ] Fold ends the game immediately
[ ] Check/Check in preflop reveals the flop
[ ] Flop contains exactly 3 community cards
[ ] Check/Check in flop goes to showdown
[ ] One full game can be played from start to finish
[ ] Reward is only -2, 0, or +2
```

---

## Task 3 — Write Environment Tests

Create:

```text
test_env.py
```

Run:

```bash
pytest test_env.py
```

Required tests:

```python
def test_reset_runs():
    env = TexasHoldemEnv()
    state, info = env.reset()
    assert state is not None
    assert isinstance(info, dict)
```

```python
def test_action_space():
    env = TexasHoldemEnv()
    assert env.action_space.n == 3
```

```python
def test_deal_two_cards_each_player():
    env = TexasHoldemEnv()
    env.reset()
    assert len(env.player_hands[0]) == 2
    assert len(env.player_hands[1]) == 2
```

```python
def test_no_duplicate_hole_cards():
    env = TexasHoldemEnv()
    env.reset()
    cards = env.player_hands[0] + env.player_hands[1]
    assert len(cards) == len(set(cards))
```

```python
def test_fold_ends_game():
    env = TexasHoldemEnv()
    env.reset()
    state, reward, terminated, truncated, info = env.step(2)

    assert terminated is True
    assert truncated is False
    assert reward in [-2, 2]
    assert info["winner"] in [0, 1]
```

```python
def test_check_check_reveals_flop():
    env = TexasHoldemEnv()
    env.reset()

    env.step(0)
    state, reward, terminated, truncated, info = env.step(0)

    assert env.phase == "flop"
    assert len(env.community_cards) == 3
    assert terminated is False
```

```python
def test_showdown_after_flop_actions():
    env = TexasHoldemEnv()
    env.reset()

    env.step(0)
    env.step(0)

    env.step(0)
    state, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert reward in [-2, 0, 2]
    assert info["winner"] in [0, 1, None]
```

```python
def test_game_can_finish():
    env = TexasHoldemEnv()
    state, info = env.reset()

    terminated = False
    reward = 0

    for _ in range(10):
        if terminated:
            break

        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert reward in [-2, 0, 2]
```

---

## Phase 1 Acceptance Criteria

Phase 1 is complete when:

```text
[ ] texas_holdenv.py exists
[ ] test_env.py exists
[ ] TexasHoldemEnv imports successfully
[ ] reset() runs without error
[ ] step() runs without error
[ ] Fold ends the game correctly
[ ] Check/Check in preflop reveals the flop
[ ] Flop has exactly 3 community cards
[ ] Check/Check in flop triggers showdown
[ ] One full game can finish
[ ] Reward is always -2, 0, or +2
[ ] pytest test_env.py passes all tests
```

---

## Important Notes for Codex

Only implement Phase 1.

Do not create or implement:

```text
q_learning_agent.py
train.py
evaluate.py
q_table.npy
rewards.npy
win_rates.npy
visualization
```

Focus only on:

```text
1. texas_holdenv.py
2. reset()
3. step()
4. flop logic
5. fold logic
6. simple showdown logic
7. test_env.py
8. fixing bugs until all tests pass
```
