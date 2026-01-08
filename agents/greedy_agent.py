"""
Greedy agent that always moves toward food using Manhattan distance
"""

import numpy as np
from .base import Agent
from env.snake_env import Direction, Action


class GreedyAgent(Agent):
    """
    Agent that greedily moves toward food.
    Uses Manhattan distance and chooses action that minimizes distance to food.
    """

    def __init__(self, action_space_size: int, env_ref=None):
        """
        Initialize greedy agent.

        Args:
            action_space_size: Number of possible actions
            env_ref: Reference to environment (to access game state)
        """
        super().__init__(action_space_size)
        self.env_ref = env_ref

    def act(self, observation: np.ndarray, **kwargs) -> int:
        """
        Choose action that moves closer to food.

        Args:
            observation: Current state observation

        Returns:
            Action that minimizes Manhattan distance to food
        """
        if self.env_ref is None:
            # Fallback to random if no env reference
            return np.random.randint(self.action_space_size)

        # Get current state
        state = self.env_ref.get_state()
        head = state['snake'][0]
        food = state['food']
        current_direction = state['direction']

        # Calculate Manhattan distance for each possible action
        best_action = Action.STRAIGHT
        best_distance = float('inf')

        for action in [Action.STRAIGHT, Action.LEFT, Action.RIGHT]:
            # Get new direction
            new_direction = self._get_new_direction(current_direction, action)

            # Get next position
            next_pos = self._get_next_position(head, new_direction)

            # Check if valid (not collision)
            if not self._is_collision(next_pos, state['snake']):
                # Calculate Manhattan distance
                distance = abs(next_pos[0] - food[0]) + abs(next_pos[1] - food[1])

                if distance < best_distance:
                    best_distance = distance
                    best_action = action

        return int(best_action)

    def _get_new_direction(self, current_direction: Direction, action: int) -> Direction:
        """Convert relative action to absolute direction"""
        if action == Action.STRAIGHT:
            return current_direction
        elif action == Action.LEFT:
            return Direction((current_direction - 1) % 4)
        elif action == Action.RIGHT:
            return Direction((current_direction + 1) % 4)

    def _get_next_position(self, pos: tuple, direction: Direction) -> tuple:
        """Get next position given current position and direction"""
        row, col = pos
        if direction == Direction.UP:
            return (row - 1, col)
        elif direction == Direction.RIGHT:
            return (row, col + 1)
        elif direction == Direction.DOWN:
            return (row + 1, col)
        elif direction == Direction.LEFT:
            return (row, col - 1)

    def _is_collision(self, pos: tuple, snake: list) -> bool:
        """Check if position results in collision"""
        row, col = pos
        board_size = self.env_ref.board_size

        # Wall collision
        if row < 0 or row >= board_size or col < 0 or col >= board_size:
            return True

        # Self collision
        if pos in snake:
            return True

        return False

    def reset(self):
        """Reset agent (no state to reset)"""
        pass
