# PROJECT_SPEC.md

# Simplified Texas Hold'em Environment

## Scope

This phase only implements the game environment:

```text
texas_holdenv.py
```

This file creates the poker game environment for the bot.

Do not implement Q-Learning, training, evaluation, or visualization in this phase.

---

## File: `texas_holdenv.py`

Status:

```text
DONE
```

Purpose:

```text
Create a simplified Texas Hold'em environment for a poker bot.
```

Required class:

```python
class SimplifiedTexasHoldemEnv(gym.Env):
```

The class must inherit from:

```python
gym.Env
```

---

## Game Rules

This is a simplified Texas Hold'em game.

There are 2 players:

```text
Player 0: Agent
Player 1: Opponent
```

Reward is calculated from the Agent's perspective.

---

## Deck

Use a 20-card deck:

```text
10, J, Q, K, A × 4 suits
```

Ranks:

```python
[10, 11, 12, 13, 14]
```

Rank mapping:

```text
11 = J
12 = Q
13 = K
14 = A
```

Suits:

```python
["C", "D", "H", "S"]
```

Each card should be represented as:

```python
(rank, suit)
```

Example:

```python
(14, "S")  # Ace of Spades
(10, "H")  # Ten of Hearts
```

Total cards:

```text
5 ranks × 4 suits = 20 cards
```

---

## Actions

The environment has 3 actions:

```text
0 = Check / Call
1 = Bet
2 = Fold
```

Declare in `__init__()`:

```python
self.action_space = spaces.Discrete(3)
```

Action meaning depends on the current betting state.

### When there is no active bet

Allowed actions:

```text
0 = Check
1 = Bet
```

Not allowed:

```text
2 = Fold
```

### When there is an active bet

Allowed actions:

```text
0 = Call
2 = Fold
```

Not allowed:

```text
1 = Bet
```

There is no raise.

---

## Observation Space

Declare an observation space in `__init__()`.

The state should include enough information for the agent to act, such as:

```text
Agent hole cards
Community cards
Current round
Current player
Pot size
Whether there is an active bet
```

The agent must not see the opponent's hole cards.

A recommended state format:

```text
[hole_1, hole_2, flop_1, flop_2, flop_3, round, current_player, pot, has_active_bet]
```

Use `-1` for unknown community cards before the flop is revealed.

Round values:

```text
0 = pre-flop
1 = flop
```

Current player values:

```text
0 = Player 0
1 = Player 1
```

Bet state:

```text
0 = no active bet
1 = active bet
```

---

## `__init__()`

The `__init__()` method must:

1. Declare `action_space`
2. Declare `observation_space`
3. Initialize the 20-card deck
4. Initialize internal game state

Required internal variables:

```python
self.deck
self.player_hands
self.community_cards
self.pot
self.round
self.current_player
self.has_active_bet
self.bettor
self.done
self.actions_this_round
```

---

## `reset()`

The `reset()` method must:

1. Shuffle the deck
2. Deal 2 hole cards to each player
3. Keep community cards hidden at the start
4. Reset the pot
5. Reset the round to pre-flop
6. Reset the current player
7. Reset betting state
8. Reset game status
9. Return the initial state

Initial game values:

```python
self.pot = 2
self.round = "preflop"
self.current_player = 0
self.has_active_bet = False
self.bettor = None
self.done = False
self.actions_this_round = []
self.community_cards = []
```

Return format:

```python
state, info
```

---

## `step(action)`

The `step(action)` method must:

1. Check if the game is already done
2. Validate the action based on the current context
3. Apply the action
4. Update pot
5. Update betting state
6. Update current player
7. Move to the next round if needed
8. Reveal the flop after pre-flop ends
9. Trigger showdown after flop ends
10. Return the result

Return format:

```python
next_state, reward, done, info
```

---

## Action Validation

### Invalid action examples

If there is no active bet:

```text
Fold is invalid.
```

Allowed:

```text
Check
Bet
```

If there is an active bet:

```text
Bet is invalid.
```

Allowed:

```text
Call
Fold
```

If the action is invalid, raise:

```python
ValueError
```

---

## Betting Logic

### Check

When action is:

```python
0
```

and there is no active bet:

```text
The player checks.
No chips are added.
Turn passes to the other player.
```

### Bet

When action is:

```python
1
```

and there is no active bet:

```text
The player bets 1 chip.
Pot increases by 1.
The betting state becomes active.
The bettor is stored.
Turn passes to the other player.
```

### Call

When action is:

```python
0
```

and there is an active bet:

```text
The player calls.
Pot increases by 1.
The active bet is cleared.
The betting round ends.
```

### Fold

When action is:

```python
2
```

and there is an active bet:

```text
The player folds.
The other player wins immediately.
The game ends.
```

---

## Round Logic

There are only 2 rounds:

```text
1. Pre-flop
2. Flop
```

There is no turn.

There is no river.

There is no raise.

Each round ends when:

```text
Both players check
```

or:

```text
One player bets and the other player calls
```

or:

```text
One player folds
```

---

## Pre-flop Logic

At the start:

```text
round = preflop
community_cards = []
current_player = 0
```

If the pre-flop round ends and nobody folds:

```text
Reveal 3 community cards.
Move to flop round.
Reset betting state.
Set current_player = 0.
```

---

## Flop Logic

During the flop round:

```text
Players can check, bet, call, or fold based on betting state.
```

If the flop round ends and nobody folds:

```text
Go to showdown.
Compare hands using hand_evaluator.py.
End the game.
```

---

## Dealing Logic

The deck contains 20 cards.

Deal:

```text
Player 0: 2 hole cards
Player 1: 2 hole cards
Community: 3 flop cards
```

Total cards used per game:

```text
7 cards
```

No duplicate cards are allowed.

---

## Winner Logic

### Fold

If a player folds:

```text
The other player wins immediately.
```

### Showdown

If no player folds:

```text
Compare both players' hands using hand_evaluator.py.
```

Each player has:

```text
2 hole cards + 3 community cards
```

The hand evaluator should determine:

```text
Player 0 wins
Player 1 wins
Draw
```

---

## Reward Logic

Reward is calculated from Player 0's perspective.

```text
Player 0 wins: +pot
Player 0 loses: -pot
Draw: 0
```

Examples:

```text
If pot = 4 and Player 0 wins  => reward = +4
If pot = 4 and Player 0 loses => reward = -4
If draw                       => reward = 0
```

---

## Expected Helper Methods

Recommended helper methods:

```python
_create_deck()
_shuffle_deck()
_deal_cards()
_card_to_id(card)
_get_obs()
_validate_action(action)
_apply_action(action)
_switch_player()
_end_round()
_reveal_flop()
_showdown()
_get_reward(winner)
```

---

## Acceptance Criteria

The environment is complete when:

```text
[ ] File texas_holdenv.py exists
[ ] Class SimplifiedTexasHoldemEnv exists
[ ] Class inherits from gym.Env
[ ] action_space has 3 actions
[ ] observation_space is declared
[ ] Deck has exactly 20 unique cards
[ ] reset() shuffles the deck
[ ] reset() deals 2 cards to each player
[ ] reset() does not reveal the flop
[ ] reset() resets pot, round, current player, and bet state
[ ] step(action) validates actions by context
[ ] Check works when there is no active bet
[ ] Bet works when there is no active bet
[ ] Call works when there is an active bet
[ ] Fold works when there is an active bet
[ ] Invalid actions raise ValueError
[ ] Pre-flop ends correctly
[ ] Flop reveals exactly 3 community cards
[ ] Flop round ends correctly
[ ] Fold ends the game immediately
[ ] Showdown uses hand_evaluator.py
[ ] Reward is +pot, -pot, or 0
[ ] No turn or river is implemented
[ ] No raise logic is implemented
```

---

## Important Constraints

Do not implement:

```text
Q-Learning
Training
Evaluation
Visualization
Turn
River
Raise
Multiple betting cycles
```

Only implement:

```text
SimplifiedTexasHoldemEnv
20-card deck
Pre-flop
Flop
Check / Call
Bet
Fold
Showdown
Reward by pot
```
