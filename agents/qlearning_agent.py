"""
Q-Learning agent with tabular Q-table
"""

import numpy as np
import pickle
from typing import Dict, Tuple
from .base import Agent


class QLearningAgent(Agent):
    """
    Tabular Q-Learning agent.

    Uses discretized feature vector as state representation.
    Best for smaller board sizes due to state space explosion.
    """

    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None
    ):
        """
        Initialize Q-Learning agent.

        Args:
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            seed: Random seed
        """
        super().__init__(action_space_size)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: dict mapping (state_key) -> array of Q-values for each action
        self.q_table: Dict[str, np.ndarray] = {}

        # Statistics
        self.episode_count = 0

        # Random number generator
        self.rng = np.random.RandomState(seed)

    def _state_to_key(self, observation: np.ndarray) -> str:
        """
        Convert observation to discrete state key.

        We discretize continuous features into bins to reduce state space.
        """
        # Discretize observation (14 features)
        discretized = []

        # Features 0-1: Direction to food (dx, dy) - 3 bins each: negative, zero, positive
        for i in range(2):
            if observation[i] < -0.1:
                discretized.append(-1)
            elif observation[i] > 0.1:
                discretized.append(1)
            else:
                discretized.append(0)

        # Features 2-4: Danger indicators (binary)
        for i in range(2, 5):
            discretized.append(int(observation[i] > 0.5))

        # Features 5-8: Current direction (one-hot) - convert to single integer
        direction = int(np.argmax(observation[5:9]))
        discretized.append(direction)

        # Features 9-12: Distance to walls - 2 bins each: close, far
        for i in range(9, 13):
            discretized.append(int(observation[i] > 0.3))

        # Feature 13: Snake length - ignore for now to reduce state space

        return str(tuple(discretized))

    def _get_q_values(self, state_key: str) -> np.ndarray:
        """Get Q-values for a state, initializing if necessary"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size, dtype=np.float32)
        return self.q_table[state_key]

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
        if self.rng.random() < epsilon:
            # Explore: random action
            return self.rng.randint(self.action_space_size)
        else:
            # Exploit: best action
            state_key = self._state_to_key(observation)
            q_values = self._get_q_values(state_key)
            return int(np.argmax(q_values))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        """
        Update Q-table using Q-learning update rule.

        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode is done
        """
        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)

        # Get current Q-value
        q_values = self._get_q_values(state_key)
        current_q = q_values[action]

        # Get max Q-value for next state
        if done:
            next_max_q = 0.0
        else:
            next_q_values = self._get_q_values(next_state_key)
            next_max_q = np.max(next_q_values)

        # Q-learning update
        target = reward + self.discount_factor * next_max_q
        q_values[action] = current_q + self.learning_rate * (target - current_q)

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def reset(self):
        """Reset episode-specific state"""
        pass

    def save(self, path: str):
        """Save Q-table to file"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_end': self.epsilon_end
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Q-Learning agent saved to {path}")
        print(f"  Q-table size: {len(self.q_table)} states")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Epsilon: {self.epsilon:.4f}")

    def load(self, path: str):
        """Load Q-table from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon_decay = data['epsilon_decay']
        self.epsilon_end = data['epsilon_end']

        print(f"Q-Learning agent loaded from {path}")
        print(f"  Q-table size: {len(self.q_table)} states")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Epsilon: {self.epsilon:.4f}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }
