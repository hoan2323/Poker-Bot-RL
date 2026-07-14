"""
config.py - Hyperparameters for NFSP Poker Bot
"""

import torch

# Network Architecture
INPUT_SIZE = 186  # State vector size
HIDDEN_LAYERS = [128, 64]
OUTPUT_SIZE = 3  # Number of actions (Check/Bet, Call/Raise, Fold)

# Training Parameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 256
TRAIN_ITERATIONS = 500000
TARGET_UPDATE_FREQ = 500  # Sync target network every N steps

# NFSP Parameters
ANTICIPATORY_PARAM = 0.25  # η (eta) - probability of using best response
TARGET_UPDATE_TAU = 0.001  # Soft update rate for target network

# Memory Parameters - SMALL for speed
RL_BUFFER_SIZE = 20000
RESERVOIR_SIZE = 20000
MIN_RL_SAMPLES = 1000  # Minimum samples before training RL
MIN_SL_SAMPLES = 2000  # Minimum samples before training SL

# Game Parameters
NUM_PLAYERS = 2
DECK_SIZE = 20
HOLE_CARDS = 2
COMMUNITY_CARDS = 5
STARTING_POT = 2
BET_SIZE = 1

# Device - Auto detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random Seed
RANDOM_SEED = 42
