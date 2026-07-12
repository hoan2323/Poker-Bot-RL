# Poker Bot NFSP - 20 Card Short Deck

Neural Fictitious Self-Play (NFSP) implementation for Short Deck Texas Hold'em.

## Setup

```bash
cd E:\document\code\poker-bot\poker-nfsp-20cards
pip install -r requirements.txt
```

## Quick Start

### 1. Test Components
```bash
python test_all.py
```

### 2. Train
```bash
python train.py
```

### 3. Evaluate
```bash
python evaluate.py
```

### 4. Play Against Bot
```bash
python play_human.py --model nfsp_agent_final.pt
```

## Project Structure

```
poker-nfsp-20cards/
├── config.py           # Hyperparameters
├── environment.py      # Game logic & state encoding
├── networks.py        # Q-Network & Policy Network
├── replay_buffer.py   # M_RL (Circular Buffer)
├── reservoir.py      # M_SL (Reservoir Sampling)
├── nfsp_agent.py     # NFSP Agent
├── train.py          # Training loop
├── evaluate.py       # Evaluation
├── aiplayer.py       # Model loader
├── play_human.py     # Human vs AI
├── requirements.txt  # Dependencies
├── README.md         # This file
└── SPEC.md           # Detailed specification
```

## Game Rules

- **Deck:** 20 cards (10, J, Q, K, A × 4 suits)
- **Players:** 2
- **Rounds:** 4 (preflop, flop, turn, river)
- **Actions:** Check/Bet, Call/Raise, Fold

## NFSP Algorithm

NFSP combines Deep Q-Learning with Fictitious Self-Play:
- **Q-Network:** Learns best response
- **Policy Network:** Learns average strategy
- **Mixture Policy:** Combines both with η (anticipatory parameter)

## References

Heinrich, J., Lanctot, M., & Silver, D. (2016). Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.
