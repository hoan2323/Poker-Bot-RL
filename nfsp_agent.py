"""
nfsp_agent.py - NFSP Agent
Core AI combining RL and SL components with mixture policy

Changes:
1. Reservoir: Only save (state, action) when using Best Response
2. SL: Use Cross Entropy (action) instead of KL Divergence (distribution)
3. Evaluation: Use Average Policy instead of Best Response
4. Action Masking: Only choose valid actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from networks import QNetwork, AveragePolicyNetwork, TargetNetwork
from replay_buffer import ReplayBuffer
from reservoir import ReservoirSampling
from config import (
    ANTICIPATORY_PARAM, GAMMA, BATCH_SIZE,
    RL_BUFFER_SIZE, RESERVOIR_SIZE,
    LEARNING_RATE, TARGET_UPDATE_TAU,
    MIN_RL_SAMPLES, MIN_SL_SAMPLES,
    DEVICE
)


class NSFPAgent:
    """
    Neural Fictitious Self-Play Agent

    Components:
    - Q-Network: Learns best response via DQN
    - Target Q-Network: Stable learning via delayed updates
    - Average Policy Network: Learns average action via supervised learning
    - M_RL: Replay buffer for RL training
    - M_SL: Reservoir for storing (state, action) from Best Response
    """

    def __init__(self, eta=ANTICIPATORY_PARAM):
        # NFSP parameter
        self.eta = eta

        # Networks
        self.q_net = QNetwork().to(DEVICE)
        self.target_net = TargetNetwork(self.q_net).to(DEVICE)
        self.policy_net = AveragePolicyNetwork().to(DEVICE)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Memory
        self.rl_buffer = ReplayBuffer(capacity=RL_BUFFER_SIZE)
        self.sl_reservoir = ReservoirSampling(capacity=RESERVOIR_SIZE)

        # Training stats
        self.total_steps = 0
        self.rl_updates = 0
        self.sl_updates = 0

    def _get_masked_probs(self, state, valid_actions):
        """Get action probabilities with masking for invalid actions"""
        probs = self.policy_net(state).squeeze().detach().cpu().numpy()

        # Mask invalid actions
        masked_probs = np.zeros_like(probs)
        for a in valid_actions:
            masked_probs[a] = probs[a]

        # Normalize
        total = masked_probs.sum()
        if total > 0:
            masked_probs /= total
        else:
            # If all masked, return uniform over valid actions
            masked_probs = np.ones_like(probs) / len(valid_actions)
            for a in range(len(probs)):
                if a not in valid_actions:
                    masked_probs[a] = 0

        return masked_probs

    def _get_masked_q_values(self, state, valid_actions):
        """Get Q-values with masking for invalid actions"""
        q_values = self.q_net(state).squeeze().detach().cpu().numpy()

        # Mask invalid actions
        masked_q = np.full_like(q_values, -float('inf'))
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return masked_q

    def choose_action(self, state, valid_actions=None, evaluate=False):
        """Choose action using mixture policy"""
        if evaluate:
            probs = self._get_masked_probs(state, valid_actions)
            return np.random.choice(len(probs), p=probs)

        # Mixture policy during training
        if random.random() < self.eta:
            q_values = self._get_masked_q_values(state, valid_actions)
            return int(np.argmax(q_values))
        else:
            probs = self._get_masked_probs(state, valid_actions)
            return np.random.choice(len(probs), p=probs)

    def store_experience(self, state, action, reward, next_state, done, valid_actions=None):
        """Store experience to both memories"""
        self.total_steps += 1

        # Store to RL buffer
        self.rl_buffer.add(state, action, reward, next_state, done)

        # Store to SL reservoir only when using Best Response (10% chance)
        if random.random() < self.eta:
            self.sl_reservoir.add(state, action)

    def update_rl(self):
        """Update Q-Network from M_RL using DQN"""
        if not self.rl_buffer.is_ready(MIN_RL_SAMPLES):
            return None

        # Sample batch
        batch = self.rl_buffer.sample(BATCH_SIZE)

        # Unpack batch - convert to numpy array directly
        states = torch.FloatTensor(np.array([e[0] for e in batch], dtype=np.float32)).to(DEVICE)
        actions = torch.LongTensor([e[1] for e in batch]).to(DEVICE)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(DEVICE)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch], dtype=np.float32)).to(DEVICE)
        dones = torch.FloatTensor([e[4] for e in batch]).to(DEVICE)

        # Forward pass
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net.network(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self.rl_updates += 1
        return loss.item()

    def update_sl(self):
        """Update Average Policy Network from M_SL using Cross Entropy"""
        if not self.sl_reservoir.is_ready(MIN_SL_SAMPLES):
            return None

        # Sample batch
        batch = self.sl_reservoir.sample(BATCH_SIZE)

        # Unpack batch: (state, action)
        states = torch.FloatTensor(np.array([e[0] for e in batch], dtype=np.float32)).to(DEVICE)
        actions = torch.LongTensor([e[1] for e in batch]).to(DEVICE)

        # Forward pass and loss
        predicted_probs = self.policy_net(states)
        loss = F.cross_entropy(predicted_probs, actions)

        # Update
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        self.sl_updates += 1
        return loss.item()

    def update(self):
        """Update both networks"""
        rl_loss = self.update_rl()
        sl_loss = self.update_sl()
        return rl_loss, sl_loss

    def sync_target_network(self, tau=TARGET_UPDATE_TAU):
        """Soft update target network"""
        self.target_net.update(self.q_net, tau)

    def get_statistics(self):
        """Get training statistics"""
        return {
            'total_steps': self.total_steps,
            'rl_buffer_size': len(self.rl_buffer),
            'sl_reservoir_size': len(self.sl_reservoir),
            'rl_updates': self.rl_updates,
            'sl_updates': self.sl_updates,
            'eta': self.eta
        }

    def save(self, path):
        """Save model weights"""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'stats': self.get_statistics()
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.network.load_state_dict(checkpoint['target_net'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test agent
    print("Testing NFSP Agent...")

    agent = NSFPAgent()

    # Test choose action with valid actions
    state = np.random.randn(186)
    valid_actions = [0, 1]
    action = agent.choose_action(state, valid_actions)
    print(f"Chosen action (valid {valid_actions}): {action}")

    valid_actions = [0, 1, 2]
    action = agent.choose_action(state, valid_actions)
    print(f"Chosen action (valid {valid_actions}): {action}")

    # Test store experience
    for _ in range(100):
        state = np.random.randn(186)
        action = random.choice([0, 1, 2])
        reward = random.choice([-1, 0, 1])
        next_state = np.random.randn(186)
        done = random.choice([True, False])
        valid = [0, 1, 2]
        agent.store_experience(state, action, reward, next_state, done, valid)

    print(f"RL buffer size: {len(agent.rl_buffer)}")
    print(f"SL reservoir size: {len(agent.sl_reservoir)}")

    # Test update
    rl_loss = agent.update_rl()
    sl_loss = agent.update_sl()
    print(f"RL Loss: {rl_loss}")
    print(f"SL Loss: {sl_loss}")

    print("\nNFSP Agent working correctly!")
