"""
Experience Replay Buffer for DQN
"""

import numpy as np
from typing import Tuple
import random


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, capacity: int, seed: int = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum size of buffer
            seed: Random seed
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
