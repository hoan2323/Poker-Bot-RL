"""
networks.py - Neural Network Architectures for NFSP
Q-Network and Average Policy Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS, LEARNING_RATE


class QNetwork(nn.Module):
    """
    Deep Q-Network for RL component
    Approximates Q(s, a) - action values
    """

    def __init__(self, hidden_layers=HIDDEN_LAYERS):
        super(QNetwork, self).__init__()

        layers = []
        input_size = INPUT_SIZE

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, OUTPUT_SIZE))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass: state -> Q-values for each action"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.network(state)

    def get_action(self, state):
        """Get greedy action (best Q-value)"""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class AveragePolicyNetwork(nn.Module):
    """
    Policy Network for SL component
    Approximates average policy π̄(s)
    """

    def __init__(self, hidden_layers=HIDDEN_LAYERS):
        super(AveragePolicyNetwork, self).__init__()

        layers = []
        input_size = INPUT_SIZE

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, OUTPUT_SIZE))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass: state -> action probabilities"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.network(state)

    def get_action(self, state):
        """Get stochastic action from policy"""
        probs = self.forward(state).squeeze()
        return np.random.choice(len(probs), p=probs.cpu().numpy())

    def get_probs(self, state):
        """Get action probabilities"""
        with torch.no_grad():
            return self.forward(state).squeeze().cpu().numpy()


class TargetNetwork(nn.Module):
    """
    Target Q-Network for stable DQN learning
    Periodically copies weights from Q-Network
    """

    def __init__(self, q_network):
        super(TargetNetwork, self).__init__()
        self.network = q_network.copy()

    def update(self, q_network, tau=0.001):
        """Soft update: θ- = τθ + (1-τ)θ-"""
        for target_param, param in zip(self.network.parameters(), q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, q_network):
        """Hard update: copy all weights"""
        self.network.load_state_dict(q_network.state_dict())


def create_networks(device="cpu"):
    """Create all networks"""
    q_net = QNetwork().to(device)
    target_net = TargetNetwork(q_net).to(device)
    policy_net = AveragePolicyNetwork().to(device)

    return q_net, target_net, policy_net


if __name__ == "__main__":
    # Test networks
    print("Testing Networks...")

    q_net = QNetwork()
    policy_net = AveragePolicyNetwork()

    # Test forward pass
    state = torch.randn(186)
    q_values = q_net(state)
    probs = policy_net(state)

    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    print(f"Action probabilities: {probs}")
    print(f"Greedy action: {q_net.get_action(state)}")
    print(f"Stochastic action: {policy_net.get_action(state)}")

    print("\nNetworks working correctly!")
