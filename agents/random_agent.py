"""
Random agent that selects actions uniformly at random
"""

import numpy as np
from .base import Agent


class RandomAgent(Agent):
    """Agent that selects actions uniformly at random"""

    def __init__(self, action_space_size: int, seed: int = None):
        """
        Initialize random agent.

        Args:
            action_space_size: Number of possible actions
            seed: Random seed for reproducibility
        """
        super().__init__(action_space_size)
        self.rng = np.random.RandomState(seed)

    def act(self, observation: np.ndarray, **kwargs) -> int:
        """
        Select random action.

        Args:
            observation: Current state observation (unused)

        Returns:
            Random action
        """
        return self.rng.randint(self.action_space_size)

    def reset(self):
        """Reset agent (no state to reset)"""
        pass
