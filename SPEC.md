# Poker Bot NFSP - Specification Document

## 1. Project Overview

**Project Name:** Poker Bot NFSP - 20 Card Short Deck

**Objective:** Build a poker AI using Neural Fictitious Self-Play (NFSP) algorithm for a simplified Texas Hold'em variant with 20 cards and 4 betting rounds.

**Algorithm:** Neural Fictitious Self-Play (NFSP) combines Deep Reinforcement Learning with Game Theory to achieve Nash Equilibrium in imperfect information games.

**Reference Paper:** Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (Heinrich, Lanctot, Silver, 2016)

---

## 2. Requirements

### 2.1 Environment Requirements

- Python 3.8+
- PyTorch (for neural networks)
- NumPy
- Gymnasium (optional, for environment structure)

### 2.2 Dependencies

```
torch>=2.0.0
numpy>=1.21.0
```

### 2.3 Hardware Requirements

- GPU recommended (but CPU works for this small game)
- Minimum 4GB RAM
- ~500MB disk space for models

---

## 3. Game Rules Specification

### 3.1 Deck Configuration

- **Deck Size:** 20 cards
- **Ranks:** 10, J, Q, K, A (5 ranks)
- **Suits:** Clubs, Diamonds, Hearts, Spades (4 suits)
- **Card Encoding:** 0-19 (rank × 4 + suit)

### 3.2 Game Structure

- **Players:** 2 (heads-up)
- **Hole Cards:** 2 per player (hidden)
- **Community Cards:** 5 total (revealed progressively)
- **Starting Ante:** 1 unit per player (total pot = 2)

### 3.3 Betting Rounds

1. **Preflop** - Before any community cards
2. **Flop** - 3 community cards revealed
3. **Turn** - 1 more community card
4. **River** - Final community card

### 3.4 Actions

| State | Valid Actions |
|-------|--------------|
| No bet yet | Check (0), Bet (1) |
| Facing bet | Call (0), Raise (1), Fold (2) |

**Action Details:**
- **Check:** Pass turn without betting (only when no bet)
- **Bet:** Wager 1 unit (first bet in round)
- **Call:** Match current bet amount
- **Raise:** Increase bet by 1 unit
- **Fold:** Discard hand, opponent wins pot

### 3.5 Betting Limits

- **Bet Size:** 1 unit
- **Raise Size:** 1 unit
- **No maximum raise limit** (but limited by pot)

### 3.6 Hand Rankings

| Rank | Name |
|------|------|
| 0 | High Card |
| 1 | One Pair |
| 2 | Two Pair |
| 3 | Three of a Kind |
| 4 | Straight |
| 5 | Flush |
| 6 | Full House |
| 7 | Four of a Kind |
| 8 | Straight Flush |

### 3.7 Game Flow

```
Round 0 (Preflop):
  Player 0 acts → Player 1 acts → If both checked/bet/call → Next round

Round 1 (Flop):
  3 cards revealed → Same betting flow

Round 2 (Turn):
  1 card revealed → Same betting flow

Round 3 (River):
  1 card revealed → Same betting flow

Terminal:
  If fold occurred → Opponent wins pot
  If river ended → Showdown → Compare hands
```

---

## 4. NFSP Algorithm Specification

### 4.1 Overview

NFSP consists of two learning components:

1. **Supervised Learning (SL):** Learns average strategy via supervised reservoir
2. **Reinforcement Learning (RL):** Learns best response via DQN

### 4.2 Architecture

```
┌────────────────────────────────────────────────────────┐
│                    NFSP Agent                           │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  Average Policy   │    │    Best Response         │  │
│  │  Network (πθ)     │    │    Q-Network (Qφ)       │  │
│  │  - Supervised    │    │    - Deep Q-Learning     │  │
│  │  - Cross-entropy │    │    - Bellman Equation   │  │
│  └────────┬─────────┘    └──────────┬───────────────┘  │
│           │                          │                  │
│           │    ┌─────────────────┐   │                  │
│           └───►│ Mixture Policy   │◄──┘                  │
│                │ π* = (1-η)πθ + ησ                      │
│                │ η = anticipatory param                  │
│                └─────────────────┘                       │
└────────────────────────────────────────────────────────┘
```

### 4.3 Memory Components

| Memory | Type | Purpose | Size |
|--------|------|---------|------|
| M_RL | Circular Buffer | RL training (DQN) | 200,000 |
| M_SL | Reservoir Sampling | SL training (Average Policy) | 2,000,000 |

### 4.4 Key Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Anticipatory | η | 0.05 | Probability of using best response |
| Discount Factor | γ | 0.99 | Future reward weight |
| Learning Rate | α | 0.01 | Gradient step size |
| Batch Size | B | 256 | Samples per update |
| Target Update | τ | 0.001 | Target network sync rate |
| Reservoir Size | | 2,000,000 | SL memory capacity |
| RL Buffer Size | | 200,000 | RL memory capacity |

### 4.5 Training Process

```
1. Initialize:
   - Q-network Qφ with random weights
   - Average policy network πθ with random weights
   - Target network Qφ- = Qφ

2. For each iteration:
   a. Self-play: Both agents play using mixture policy
   b. Store experiences:
      - RL Buffer: (s, a, r, s', done) for each step
      - Reservoir: (s, π*) for selected states
   c. Update Q-network via DQN:
      - Sample batch from M_RL
      - Compute target: y = r + γ max Qφ-(s', a')
      - Gradient descent on (Qφ(s,a) - y)²
   d. Update Average Policy:
      - Sample batch from M_SL
      - Cross-entropy loss between πθ(s) and stored policies
   e. Periodically sync target network:
      - φ- ← τφ + (1-τ)φ-

3. After training:
   - Use πθ (average policy) for final strategy
```

---

## 5. Project Structure

### 5.1 File List

```
poker-nfsp-20cards/
├── config.py             # Hyperparameters
├── environment.py        # Game logic & state encoding
├── networks.py          # Neural network architectures
├── replay_buffer.py     # M_RL (Circular Buffer)
├── reservoir.py         # M_SL (Reservoir Sampling)
├── nfsp_agent.py       # NFSP Agent
├── train.py            # Self-play training loop
├── evaluate.py          # Evaluation scripts
├── aiplayer.py         # Model loader
└── play_human.py       # Human vs AI interface
```

---

## 6. File Specifications

### 6.1 config.py

**Purpose:** Centralize all hyperparameters for easy tuning.

**Contents:**
- Neural network architecture: hidden layers, neuron counts
- Training parameters: learning rate, batch size, gamma
- Memory parameters: buffer sizes, reservoir capacity
- NFSP specific: anticipatory parameter (eta)
- Evaluation settings: number of games, opponent types

**Example:**
```python
# Network
HIDDEN_LAYERS = [128, 64]
LEARNING_RATE = 0.001

# Memory
RL_BUFFER_SIZE = 200000
RESERVOIR_SIZE = 2000000
BATCH_SIZE = 256

# NFSP
ANTICIPATORY_PARAM = 0.05
GAMMA = 0.99
TARGET_UPDATE_TAU = 0.001

# Training
TRAIN_ITERATIONS = 1000000
EVAL_FREQUENCY = 10000
```

**Functions:** None (module only)

---

### 6.2 environment.py

**Purpose:** Game environment handling all game logic and state representation.

**Classes:**
- `ShortDeckPokerEnv` - Main environment class

**Key Methods:**
- `reset(starting_player)` - Initialize new game, return state
- `step(action)` - Execute action, return (next_state, reward, done, info)
- `get_state(player)` - Get 160-bit state encoding for player
- `get_valid_actions()` - Return valid actions for current state
- `get_reward(player)` - Get reward for player at terminal state

**State Encoding (160-bit):**
```
Bit Layout:
[0-19]   : Player 0 hole card 1 (one-hot)
[20-39]  : Player 0 hole card 2 (one-hot)
[40-59]  : Player 1 hole card 1 (one-hot, hidden = 0)
[60-79]  : Player 1 hole card 2 (one-hot, hidden = 0)
[80-99]  : Community card 1 (one-hot)
[100-119]: Community card 2 (one-hot)
[120-139]: Community card 3 (one-hot)
[140-159]: Community card 4 (one-hot)
[160-179]: Community card 5 (one-hot)
[180]    : Pot size (binary, 0/1 for simplicity)
[181-182]: Round index (0-3)
[183]    : Current player (0 or 1)
[184]    : Has active bet (0 or 1)
[185]    : Bet facing (0 or 1)
```

**Functions:**
- `evaluate_hand(cards)` - Return hand rank and kicker info
- `compare_hands(hand1, hand2)` - Return winner (1, -1, or 0)
- `card_to_rank(card)` - Extract rank from card index
- `card_to_suit(card)` - Extract suit from card index

---

### 6.3 networks.py

**Purpose:** Define neural network architectures for Q-function and average policy.

**Classes:**
- `QNetwork` - Deep Q-Network for RL component
- `AveragePolicyNetwork` - Policy network for SL component

**QNetwork Architecture:**
```
Input:  186 features (state vector)
   ↓
Hidden: 128 neurons, ReLU
   ↓
Hidden: 64 neurons, ReLU
   ↓
Output: 3 neurons (Q-values for each action)
```

**AveragePolicyNetwork Architecture:**
```
Input:  186 features (state vector)
   ↓
Hidden: 128 neurons, ReLU
   ↓
Hidden: 64 neurons, ReLU
   ↓
Output: 3 neurons (softmax for action probabilities)
```

**Methods:**
- `forward(state)` - Compute output given state
- `get_action(state)` - Get greedy action from Q-values
- `get_policy(state)` - Get action probabilities from policy network

---

### 6.4 replay_buffer.py

**Purpose:** Circular buffer for storing RL experiences (M_RL).

**Class:** `ReplayBuffer`

**Data Structure:** Ring buffer with fixed capacity

**Stored Experience Format:**
```python
{
    'state': np.array,      # Current state (186 features)
    'action': int,          # Action taken (0, 1, or 2)
    'reward': float,        # Reward received
    'next_state': np.array, # Resulting state
    'done': bool            # Episode end flag
}
```

**Methods:**
- `add(state, action, reward, next_state, done)` - Add experience
- `sample(batch_size)` - Random sample for training
- `__len__()` - Current number of experiences
- `is_ready(batch_size)` - Check if enough samples available

---

### 6.5 reservoir.py

**Purpose:** Reservoir sampling for storing SL experiences (M_SL).

**Class:** `ReservoirSampling`

**Data Structure:** Fixed-size reservoir with random replacement

**Stored Experience Format:**
```python
{
    'state': np.array,      # State at decision point
    'policy': np.array      # Average policy π̄ from anticipatory agent
}
```

**Methods:**
- `add(state, policy)` - Add experience (random replacement)
- `sample(batch_size)` - Random sample for training
- `__len__()` - Current number of samples
- `is_ready(batch_size)` - Check if enough samples available

---

### 6.6 nfsp_agent.py

**Purpose:** NFSP agent combining RL and SL components with mixture policy.

**Class:** `NSFPAgent`

**Components:**
- Q-Network (φ) for best response
- Target Q-Network (φ̄) for stable learning
- Average Policy Network (θ)
- RL Replay Buffer (M_RL)
- SL Reservoir (M_SL)

**Key Methods:**
- `choose_action(state, evaluate=False)` - Select action using mixture policy
- `store_experience(state, action, reward, next_state, done)` - Store in memories
- `update_rl()` - Update Q-network from M_RL
- `update_sl()` - Update policy network from M_SL
- `sync_target_network()` - Update target network weights
- `save(path)` - Save model weights
- `load(path)` - Load model weights

**Mixture Policy:**
```python
def get_mixture_action(state):
    if random() < eta:  # eta = anticipatory param
        # Best response (from Q-network)
        action = q_network.get_greedy_action(state)
    else:
        # Average policy (from SL network)
        action = policy_network.get_stochastic_action(state)
    return action
```

---

### 6.7 train.py

**Purpose:** Training loop for NFSP agents via self-play.

**Functions:**
- `train()` - Main training loop
- `play_episode(agent1, agent2)` - Single self-play episode
- `update_agents(agent1, agent2)` - Update both agents
- `log_progress(iteration, metrics)` - Logging and checkpointing

**Training Loop:**
```python
def train():
    agent1 = NSFPAgent()
    agent2 = NSFPAgent()

    for iteration in range(TRAIN_ITERATIONS):
        # Self-play episode
        play_episode(agent1, agent2)

        # Update both agents
        if iteration % UPDATE_FREQUENCY == 0:
            agent1.update_rl()
            agent1.update_sl()
            agent2.update_rl()
            agent2.update_sl()

        # Evaluation
        if iteration % EVAL_FREQUENCY == 0:
            evaluate(agent1, agent2)
            log_progress(iteration)

    # Save final models
    agent1.save("agent1_final.pt")
    agent2.save("agent2_final.pt")
```

---

### 6.8 evaluate.py

**Purpose:** Evaluate trained agents against baseline opponents.

**Functions:**
- `evaluate_vs_random(agent, n_games)` - Agent vs random player
- `evaluate_vs_call_station(agent, n_games)` - Agent vs call station
- `evaluate_vs_heuristic(agent, n_games)` - Agent vs heuristic bot
- `evaluate_self_play(agent1, agent2, n_games)` - Agent vs agent
- `print_results(results)` - Display evaluation metrics

**Baseline Opponents:**
- **Random:** Chooses random valid action
- **Call Station:** Always calls, never folds
- **Tight:** Only plays strong hands
- **Aggressive:** Always bets when possible
- **Heuristic:** Rule-based using hand strength

**Metrics:**
- Win rate
- Loss rate
- Tie rate
- Average reward
- Average pot size

---

### 6.9 aiplayer.py

**Purpose:** Interface for loading trained models and playing against them.

**Class:** `AIPlayer`

**Methods:**
- `load_model(path)` - Load saved NFSP agent
- `set_evaluate_mode()` - Disable exploration (epsilon = 0)
- `get_action(state)` - Get action from agent
- `reset()` - Reset internal state

**Usage:**
```python
from aiplayer import AIPlayer

ai = AIPlayer()
ai.load_model("agent_final.pt")
ai.set_evaluate_mode()

action = ai.get_action(game_state)
```

---

### 6.10 play_human.py

**Purpose:** Human vs AI interface for testing and demonstration.

**Interface Types:**
- Terminal (text-based, simple)
- Pygame (graphical, optional)

**Terminal Interface:**
```
Pot: 5 | Round: 2 (Flop)
Board: [QH, JD, 10C]
Your Hand: [AS, KS]
Valid Actions: [0] Check, [1] Bet

Your action: 1
You bet 1 unit.
```

**Key Functions:**
- `display_game_state(env, player)` - Show current game
- `get_human_input()` - Get player's action choice
- `play_human_vs_ai()` - Main game loop
- `play_continue()` - Ask to play again

---

## 7. System Flow

### 7.1 Training Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING FLOW                             │
│                                                               │
│  ┌─────────────┐                                             │
│  │   START     │                                             │
│  └──────┬──────┘                                             │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                         │
│  │ Initialize 2     │                                         │
│  │ NFSP Agents     │                                         │
│  └──────┬──────────┘                                         │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                         │
│  │ Self-Play       │◄──────────────────────────────────┐    │
│  │ Episode         │                                    │    │
│  └──────┬──────────┘                                    │    │
│         │                                                 │    │
│         ▼                                                 │    │
│  ┌─────────────────┐                                      │    │
│  │ Store to M_RL   │◄── RL Experience                   │    │
│  │ & M_SL          │◄── SL Experience                    │    │
│  └──────┬──────────┘                                      │    │
│         │                                                 │    │
│         ▼                                                 │    │
│  ┌─────────────────┐                                      │    │
│  │ Update Networks │◄── Every UPDATE_FREQ iterations     │    │
│  │ (RL + SL)       │                                      │    │
│  └──────┬──────────┘                                      │    │
│         │                                                 │    │
│         ▼                                                 │    │
│  ┌─────────────────┐                                      │    │
│  │ Evaluate        │◄── Every EVAL_FREQ iterations        │    │
│  │ & Log           │                                      │    │
│  └──────┬──────────┘                                      │    │
│         │                                                 │    │
│         │            ┌─────────────────┐                  │    │
│         └───────────►│ Iteration <     │──Yes──────────►│    │
│                      │ MAX?            │                  │    │
│                      └────────┬────────┘                  │    │
│                               │ No                        │    │
│                               ▼                           │    │
│                      ┌─────────────────┐                  │    │
│                      │ Save Models     │                  │    │
│                      └──────┬──────────┘                  │    │
│                             │                              │    │
│                             ▼                              │    │
│                      ┌─────────────────┐                   │    │
│                      │      END        │                   │    │
│                      └─────────────────┘                   │    │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Self-Play Episode Flow

```
┌─────────────────────────────────────────────────┐
│              SELF-PLAY EPISODE                   │
│                                                  │
│  ┌───────────┐                                   │
│  │ Reset Env │                                   │
│  └─────┬─────┘                                   │
│        │                                         │
│        ▼                                         │
│  ┌───────────┐     ┌─────────────┐             │
│  │ Agent 1    │────►│ Choose A1    │             │
│  │ acts       │     │ (Mixture)   │             │
│  └─────┬─────┘     └──────┬──────┘             │
│        │                   │                      │
│        │           ┌───────▼───────┐            │
│        │           │ Store to M_RL │            │
│        │           │ Store to M_SL │            │
│        │           └───────────────┘            │
│        │                   │                      │
│        ▼                   ▼                      │
│  ┌───────────┐     ┌─────────────┐             │
│  │ Env Step  │     │ Agent 2     │             │
│  │ A1        │     │ acts        │             │
│  └─────┬─────┘     └──────┬──────┘             │
│        │                   │                      │
│        │                   ▼                      │
│        │           ┌─────────────┐             │
│        │           │ Env Step    │             │
│        │           │ A2          │             │
│        │           └──────┬──────┘             │
│        │                  │                      │
│        │                  ▼                      │
│        │           ┌─────────────┐             │
│        │           │ Store to M_RL│            │
│        │           │ Store to M_SL│            │
│        │           └──────┬──────┘             │
│        │                  │                      │
│        └──────────────────┴────►┌───────────────┐
│                                 │ Episode Done?  │
│                                 └───────┬───────┘
│                                         │ No
│                                         └────► (loop)
```

### 7.3 Evaluation Flow

```
┌─────────────────────────────────────────────────┐
│              EVALUATION FLOW                     │
│                                                  │
│  ┌───────────┐                                   │
│  │ Load      │                                   │
│  │ Trained   │                                   │
│  │ Agent     │                                   │
│  └─────┬─────┘                                   │
│        │                                         │
│        ▼                                         │
│  ┌───────────────────┐                          │
│  │ For each opponent │                          │
│  │ (Random, Heuristic│                          │
│  │  Call Station...) │                          │
│  └─────┬─────────────┘                          │
│        │                                         │
│        ▼                                         │
│  ┌───────────────────┐                          │
│  │ Play N games      │◄────────────────────┐  │
│  │ (Agent vs Opp)    │                      │  │
│  └─────┬─────────────┘                      │  │
│        │                                    │  │
│        ▼                                    │  │
│  ┌───────────────────┐     ┌─────────────┐ │  │
│  │ Record Results    │────►│ Game < N?   │ │  │
│  │ (Win/Loss/Tie)   │     │ (loop)      │─┘  │
│  └───────────────────┘     └─────────────┘    │
│        │                                          │
│        ▼                                          │
│  ┌───────────────────┐                           │
│  │ Compute Metrics   │                           │
│  │ (WinRate, etc)   │                           │
│  └───────────────────┘                           │
│        │                                         │
│        ▼                                         │
│  ┌───────────────────┐                          │
│  │ Print/Save Report │                           │
│  └───────────────────┘                          │
└─────────────────────────────────────────────────┘
```

---

## 8. Implementation Checklist

### Phase 1: Environment
- [ ] Implement card representation
- [ ] Implement hand evaluation (bitmask)
- [ ] Implement betting logic
- [ ] Implement state encoding (160-bit)
- [ ] Test environment

### Phase 2: Memory
- [ ] Implement ReplayBuffer
- [ ] Implement ReservoirSampling
- [ ] Test memory operations

### Phase 3: Networks
- [ ] Implement QNetwork
- [ ] Implement AveragePolicyNetwork
- [ ] Test forward passes

### Phase 4: Agent
- [ ] Implement NSFPAgent
- [ ] Implement mixture policy
- [ ] Implement RL update
- [ ] Implement SL update
- [ ] Test agent

### Phase 5: Training
- [ ] Implement training loop
- [ ] Add logging
- [ ] Add checkpoints
- [ ] Train agents

### Phase 6: Evaluation
- [ ] Implement baseline opponents
- [ ] Implement evaluation functions
- [ ] Run evaluations
- [ ] Generate reports

### Phase 7: Play Interface
- [ ] Implement AIPlayer
- [ ] Implement terminal interface
- [ ] Test human vs AI

---

## 9. Expected Outcomes

### 9.1 Training Metrics
- Self-play win rate should converge to ~50% (vs itself)
- Win rate vs random: >70%
- Win rate vs call station: >60%
- Win rate vs tight: ~50%

### 9.2 Convergence Indicators
- Q-values stabilize
- Policy entropy decreases
- Memory buffers fill
- Evaluation metrics plateau

### 9.3 Demo Capabilities
- Human can play against trained bot
- Bot demonstrates strategic behavior:
  - Value betting with strong hands
  - Bluffing with weak hands
  - Slow play (check-raise)
  - Folding marginal hands

---

## 10. References

1. Heinrich, J., Lanctot, M., & Silver, D. (2016). Deep Reinforcement Learning from Self-Play in Imperfect-Information Games. arXiv:1611.01741.

2. Heinrich, J., & Silver, D. (2016). Deep Reinforcement Learning from Self-Play in Imperfect-Information Games. NIPS 2016.

3. OpenAI Baselines: https://github.com/openai/baselines
