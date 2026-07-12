"""
config.py - Hyperparameters for NFSP Poker Bot
"""

# Network Architecture
INPUT_SIZE = 186  # State vector size
HIDDEN_LAYERS = [128, 64]
OUTPUT_SIZE = 3  # Number of actions (Check/Bet, Call/Raise, Fold)

# Training Parameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 256
TRAIN_ITERATIONS = 1000000
UPDATE_FREQUENCY = 1  # Update after each step
EVAL_FREQUENCY = 10000

# NFSP Parameters
ANTICIPATORY_PARAM = 0.05  # η (eta) - probability of using best response
TARGET_UPDATE_TAU = 0.001  # Soft update rate for target network

# Memory Parameters
RL_BUFFER_SIZE = 200000
RESERVOIR_SIZE = 2000000
MIN_RL_SAMPLES = 1000  # Minimum samples before training RL
MIN_SL_SAMPLES = 5000  # Minimum samples before training SL

# Evaluation Parameters
EVAL_GAMES = 10000
EVAL_BOT_GAMES = 5000

# Game Parameters
NUM_PLAYERS = 2
DECK_SIZE = 20
HOLE_CARDS = 2
COMMUNITY_CARDS = 5
STARTING_POT = 2  # 1 ante each
BET_SIZE = 1

# Device
DEVICE = "cpu"  # "cuda" if GPU available

# Check for CUDA
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
except:
    pass

# Random Seed
RANDOM_SEED = 42
