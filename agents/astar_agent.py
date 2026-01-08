"""
A* pathfinding agent with obstacle awareness
"""

import numpy as np
from typing import List, Tuple, Optional
import heapq
from .base import Agent
from env.snake_env import Direction, Action


class AStarAgent(Agent):
    """
    Agent that uses A* pathfinding to navigate to food.
    Considers obstacles (walls and snake body).
    """

    def __init__(self, action_space_size: int, env_ref=None):
        """
        Initialize A* agent.

        Args:
            action_space_size: Number of possible actions
            env_ref: Reference to environment (to access game state)
        """
        super().__init__(action_space_size)
        self.env_ref = env_ref

    def act(self, observation: np.ndarray, **kwargs) -> int:
        """
        Choose action using A* pathfinding.

        Args:
            observation: Current state observation

        Returns:
            Action from A* path, or random if no path found
        """
        if self.env_ref is None:
            return np.random.randint(self.action_space_size)

        # Get current state
        state = self.env_ref.get_state()
        head = state['snake'][0]
        food = state['food']
        current_direction = state['direction']

        # Find path using A*
        path = self._find_path(head, food, state['snake'])

        if path and len(path) > 1:
            # Get next position from path
            next_pos = path[1]

            # Determine which action leads to next_pos
            action = self._position_to_action(head, next_pos, current_direction)
            return int(action)
        else:
            # No path found, try to move safely
            return self._safe_random_action(state)

    def _find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        snake: List[Tuple[int, int]]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to goal using A*.

        Args:
            start: Starting position
            goal: Goal position
            snake: Snake body positions (obstacles)

        Returns:
            List of positions from start to goal, or None if no path
        """
        board_size = self.env_ref.board_size

        # Priority queue: (f_score, counter, position, path)
        counter = 0
        open_set = [(0, counter, start, [start])]
        closed_set = set()

        while open_set:
            f_score, _, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # Explore neighbors
            for next_pos in self._get_neighbors(current, board_size, snake):
                if next_pos in closed_set:
                    continue

                new_path = path + [next_pos]
                g_score = len(new_path) - 1
                h_score = self._manhattan_distance(next_pos, goal)
                f_score = g_score + h_score

                counter += 1
                heapq.heappush(open_set, (f_score, counter, next_pos, new_path))

        return None  # No path found

    def _get_neighbors(
        self,
        pos: Tuple[int, int],
        board_size: int,
        snake: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        row, col = pos
        neighbors = [
            (row - 1, col),  # Up
            (row, col + 1),  # Right
            (row + 1, col),  # Down
            (row, col - 1)   # Left
        ]

        valid_neighbors = []
        for neighbor in neighbors:
            r, c = neighbor
            # Check bounds
            if 0 <= r < board_size and 0 <= c < board_size:
                # Check not in snake body (allow tail since it will move)
                if neighbor not in snake[:-1]:
                    valid_neighbors.append(neighbor)

        return valid_neighbors

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _position_to_action(
        self,
        current: Tuple[int, int],
        next_pos: Tuple[int, int],
        current_direction: Direction
    ) -> Action:
        """
        Determine which relative action leads from current to next_pos.

        Args:
            current: Current position
            next_pos: Desired next position
            current_direction: Current heading direction

        Returns:
            Relative action (STRAIGHT, LEFT, or RIGHT)
        """
        # Determine absolute direction to next_pos
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]

        if dr == -1:
            target_direction = Direction.UP
        elif dr == 1:
            target_direction = Direction.DOWN
        elif dc == 1:
            target_direction = Direction.RIGHT
        elif dc == -1:
            target_direction = Direction.LEFT
        else:
            return Action.STRAIGHT

        # Convert to relative action
        diff = (target_direction - current_direction) % 4

        if diff == 0:
            return Action.STRAIGHT
        elif diff == 1:
            return Action.RIGHT
        elif diff == 3:
            return Action.LEFT
        else:
            # 180-degree turn (shouldn't happen with proper pathfinding)
            return Action.STRAIGHT

    def _safe_random_action(self, state: dict) -> int:
        """
        Choose a random action that doesn't immediately cause collision.
        Fallback when no path is found.
        """
        head = state['snake'][0]
        current_direction = state['direction']
        snake = state['snake']

        safe_actions = []
        for action in [Action.STRAIGHT, Action.LEFT, Action.RIGHT]:
            new_direction = self._get_new_direction(current_direction, action)
            next_pos = self._get_next_position(head, new_direction)

            if not self._is_collision(next_pos, snake):
                safe_actions.append(action)

        if safe_actions:
            return int(np.random.choice(safe_actions))
        else:
            # No safe action (imminent death)
            return int(Action.STRAIGHT)

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
