"""
replay_buffer.py - Circular Buffer for RL Experiences (M_RL)
Stores experiences for Deep Q-Learning
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Circular buffer for storing RL experiences
    M_RL in NFSP paper
    """

    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        Experience: (state, action, reward, next_state, done)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Random sample from buffer
        Returns: list of experiences
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

    def save(self, path):
        """Save buffer to file"""
        np.save(path, np.array(self.buffer, dtype=object))

    def load(self, path):
        """Load buffer from file"""
        data = np.load(path, allow_pickle=True)
        self.buffer = deque(data.tolist(), maxlen=self.capacity)


if __name__ == "__main__":
    # Test replay buffer
    buffer = ReplayBuffer(capacity=100)

    # Add some experiences
    for i in range(50):
        state = np.random.randn(186)
        action = random.choice([0, 1, 2])
        reward = random.choice([-1, 0, 1])
        next_state = np.random.randn(186)
        done = random.choice([True, False])

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")
    print(f"Is ready for batch 32: {buffer.is_ready(32)}")

    # Sample
    batch = buffer.sample(8)
    print(f"Sampled batch size: {len(batch)}")

    print("\nReplay buffer working correctly!")
