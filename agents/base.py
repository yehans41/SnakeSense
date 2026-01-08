"""
Base agent interface for all agents
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class Agent(ABC):
    """Base class for all agents"""

    def __init__(self, action_space_size: int):
        """
        Initialize agent.

        Args:
            action_space_size: Number of possible actions
        """
        self.action_space_size = action_space_size

    @abstractmethod
    def act(self, observation: np.ndarray, **kwargs) -> int:
        """
        Choose an action given an observation.

        Args:
            observation: Current state observation
            **kwargs: Additional arguments (e.g., epsilon for exploration)

        Returns:
            action: Action to take
        """
        pass

    def reset(self):
        """Reset agent state (if needed)"""
        pass

    def update(self, *args, **kwargs):
        """Update agent (for learning agents)"""
        pass

    def save(self, path: str):
        """Save agent to file"""
        pass

    def load(self, path: str):
        """Load agent from file"""
        pass
