# Poker Bot RL — Full Texas Hold'em with Q-Learning

## Project Overview

This project builds a complete 2-player Texas Hold'em Poker bot using tabular Q-Learning.

Implemented parts:

- Poker environment
- Hand evaluator
- Q-Learning agent
- Training script
- Evaluation script
- Unit tests
- Terminal commands

Uses 20-card short deck: 10, J, Q, K, A across 4 suits.

---

## Project Structure

```text
poker-bot/
├── texas_holdenv.py
├── hand_evaluator.py
├── q_learning_agent.py
├── train.py
├── evaluate.py
├── test_env.py
├── test_hand_evaluator.py
├── test_agent.py
├── requirements.txt
├── README.md
├── q_table.npy          # generated after training
├── rewards.npy          # generated after training
└── win_rates.npy        # generated after training
```

---

## Requirements

- Python 3.10+
- gymnasium
- numpy
- pytest
- matplotlib

Install:

```bash
pip install -r requirements.txt
```

---

## System Flow

### 1. `hand_evaluator.py`

Handles poker hand ranking and comparison.

Functions:

- `evaluate_five_card_hand(cards)`
- `evaluate_best_hand(cards)`
- `compare_hands(player0_cards, player1_cards)`

Supported rankings:

- High Card
- One Pair
- Two Pair
- Three of a Kind
- Straight
- Flush
- Full House
- Four of a Kind
- Straight Flush
- Ace-low straight

### 2. `texas_holdenv.py`

Implements full 2-player Texas Hold'em environment.

Features:

- 20-card short deck
- 2 hole cards per player
- 5 community cards
- Rounds:
  - Preflop
  - Flop
  - Turn
  - River
  - Showdown
- Actions:
  - `0 = Check / Call`
  - `1 = Bet`
  - `2 = Fold`
- Gymnasium-compatible API:
  - `reset(seed=None, options=None)`
  - `step(action)`

Observation format:

```text
[
  hole_1,
  hole_2,
  community_1,
  community_2,
  community_3,
  community_4,
  community_5,
  round,
  current_player,
  pot,
  has_active_bet
]
```

### 3. `q_learning_agent.py`

Implements compact tabular Q-learning.

State abstraction:

```text
(
  made_hand_rank,
  best_current_hand_rank,
  pair_flag,
  two_pair_flag,
  trips_flag,
  flush_draw,
  straight_draw,
  top_pair,
  overpair,
  kicker_bucket,
  pot_bucket,
  has_active_bet,
  round
)
```

Pot buckets:

```text
0 = 2
1 = 3-4
2 = 5-6
3 = 7+
```

Agent behavior:

- epsilon-greedy action selection
- valid actions only
- dictionary-based Q-table
- save/load with NumPy

### 4. `train.py`

Trains Player 0 against a weighted opponent mix:

```text
40% heuristic
30% call_station
20% random
10% always_bet
```

Training uses honest terminal environment rewards: `+pot`, `-pot`, or `0`.
A tiny Player 0-only shaping reward is added for pot-risk learning:

```text
+0.05 for folding weak hands to a bet in pots >= 5
-0.05 for calling weak hands against a bet in pots >= 5
```

Training flow:

1. Reset environment
2. Check current acting player
3. Agent acts if Player 0
4. Weighted opponent action if Player 1
5. Step environment
6. Update Q-table only when Player 0 acted
7. Track rewards and win rates
8. Decay epsilon
9. Save outputs:
   - `q_table.npy`
   - `rewards.npy`
   - `win_rates.npy`

### 5. `evaluate.py`

Loads trained Q-table and evaluates agent against random opponent.

Evaluation output:

```text
Evaluation Results
Total Games: 5000
Wins: ...
Losses: ...
Draws: ...
Win Rate: ...
Average Reward: ...
```

### 6. Tests

Files:

- `test_hand_evaluator.py`
- `test_env.py`
- `test_agent.py`

Coverage:

- hand ranking correctness
- best 5-card selection from 7 cards
- environment reset/step behavior
- valid action enforcement
- showdown flow
- reproducible seeding
- Q-learning state/action/update/save/load behavior

---

## Game Rules in This Project

Implemented:

- 2-player Texas Hold'em
- 20-card short deck
- Check / Call
- Bet
- Fold
- Preflop
- Flop
- Turn
- River
- Showdown
- Q-Learning
- Random opponent

Not implemented:

- Deep Learning
- Neural Networks
- DQN
- External poker libraries
- Multiple opponents
- Side pots
- All-in
- Raise
- Complex betting sizes
- Advanced bluffing logic
- Advanced opponent modeling

---

## Terminal Commands

### Create virtual environment

```bash
python -m venv .venv
```

### Activate on Windows

```bash
.venv\Scripts\activate
```

### Activate on macOS/Linux

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run all tests

```bash
pytest -s -v
```

### Run environment tests

```bash
pytest -s -v test_env.py
```

### Run hand evaluator tests

```bash
pytest -s -v test_hand_evaluator.py
```

### Run agent tests

```bash
pytest -s -v test_agent.py
```

### Train agent

```bash
python train.py
```

Expected generated files:

```text
q_table.npy
rewards.npy
win_rates.npy
```

### Evaluate agent

```bash
python evaluate.py
```

---

## Acceptance Checklist

- [x] `texas_holdenv.py` exists
- [x] `hand_evaluator.py` exists
- [x] `q_learning_agent.py` exists
- [x] `train.py` exists
- [x] `evaluate.py` exists
- [x] `test_env.py` exists
- [x] `test_hand_evaluator.py` exists
- [x] `requirements.txt` exists
- [x] Environment uses 20 unique short-deck cards
- [x] Environment supports preflop, flop, turn, river, showdown
- [x] Environment validates actions by context
- [x] Environment exposes `get_valid_actions()`
- [x] Environment uses Gymnasium-compatible step return
- [x] Hand evaluator supports all standard poker hand rankings
- [x] Hand evaluator picks best 5-card hand from 7 cards
- [x] Q-learning agent uses compact state abstraction
- [x] Q-learning agent never selects invalid actions
- [x] Training updates only Player 0
- [ ] Training creates `q_table.npy`
- [ ] Training creates `rewards.npy`
- [ ] Training creates `win_rates.npy`
- [ ] Evaluation loads `q_table.npy`
- [ ] `pytest -s -v` passes
- [ ] `python train.py` runs successfully
- [ ] `python evaluate.py` runs successfully