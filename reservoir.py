"""
reservoir.py - Reservoir Sampling for SL Experiences (M_SL)
Stores experiences for supervised learning of average policy

Changed: Now stores (state, action) instead of (state, policy)
Used for Cross Entropy loss to learn Best Response actions
"""

import numpy as np
import random


class ReservoirSampling:
    """
    Reservoir sampling for storing SL experiences
    M_SL in NFSP paper
    Maintains fixed-size sample with uniform probability
    """

    def __init__(self, capacity=2000000):
        self.capacity = capacity
        self.buffer = []
        self.count = 0  # Total items added

    def add(self, state, action):
        """
        Add experience using reservoir sampling
        """
        self.count += 1

        if len(self.buffer) < self.capacity:
            # Buffer not full - add new item (no copy needed)
            self.buffer.append((state, action))
        else:
            # Buffer full - random replacement
            j = random.randint(0, self.count - 1)
            if j < self.capacity:
                self.buffer[j] = (state, action)

    def sample(self, batch_size):
        """
        Random sample from reservoir
        Returns: list of (state, action) tuples
        """
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if reservoir has enough samples"""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear reservoir"""
        self.buffer = []
        self.count = 0

    def save(self, path):
        """Save reservoir to file"""
        data = {
            'buffer': self.buffer,
            'count': self.count,
            'capacity': self.capacity
        }
        np.save(path, data, allow_pickle=True)

    def load(self, path):
        """Load reservoir from file"""
        data = np.load(path, allow_pickle=True).item()
        self.buffer = data['buffer']
        self.count = data['count']
        self.capacity = data['capacity']


if __name__ == "__main__":
    # Test reservoir
    reservoir = ReservoirSampling(capacity=1000)

    # Add many samples (state, action)
    for i in range(5000):
        state = np.random.randn(186)
        action = random.choice([0, 1, 2])
        reservoir.add(state, action)

    print(f"Total added: {reservoir.count}")
    print(f"Buffer size: {len(reservoir)}")
    print(f"Is ready for batch 64: {reservoir.is_ready(64)}")

    # Sample
    batch = reservoir.sample(8)
    print(f"Sampled batch size: {len(batch)}")
    print(f"First sample: state shape={batch[0][0].shape}, action={batch[0][1]}")

    print("\nReservoir working correctly!")
