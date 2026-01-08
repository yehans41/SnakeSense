"""
Deep Q-Network (DQN) agent using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Agent
from rl.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Q-Network for DQN"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class DQNAgent(Agent):
    """
    Deep Q-Network agent with experience replay and target network.
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        learning_rate: float = 0.0005,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update_frequency: int = 100,
        hidden_dims: List[int] = [128, 128],
        seed: int = None,
        device: str = None
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_space_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            target_update_frequency: Steps between target network updates
            hidden_dims: Hidden layer dimensions
            seed: Random seed
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__(action_space_size)

        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create networks
        self.q_network = QNetwork(state_dim, action_space_size, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_space_size, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(memory_size, seed=seed)

        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.losses = []

    def act(self, observation: np.ndarray, epsilon: float = None) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            observation: Current state observation
            epsilon: Override epsilon for this action (optional)

        Returns:
            Action to take
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.action_space_size)
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return int(q_values.argmax().item())

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        """
        Store experience and train if enough samples.

        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode is done
        """
        # Store experience
        self.memory.push(observation, action, reward, next_observation, done)

        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            loss = self._train_step()
            self.losses.append(loss)

            # Update target network
            if self.training_step % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_step(self) -> float:
        """
        Perform one training step.

        Returns:
            Loss value
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def reset(self):
        """Reset episode-specific state"""
        pass

    def save(self, path: str):
        """Save agent to file"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'config': {
                'state_dim': self.state_dim,
                'action_space_size': self.action_space_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_end': self.epsilon_end,
                'batch_size': self.batch_size,
                'target_update_frequency': self.target_update_frequency
            }
        }
        torch.save(checkpoint, path)
        print(f"DQN agent saved to {path}")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Training steps: {self.training_step}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Memory size: {len(self.memory)}")

    def load(self, path: str):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']

        print(f"DQN agent loaded from {path}")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Training steps: {self.training_step}")
        print(f"  Epsilon: {self.epsilon:.4f}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
        return {
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'avg_loss': avg_loss
        }
