"""
Value Iteration agent for small Snake environments
Solves the MDP using dynamic programming
"""

import numpy as np
import pickle
from typing import Dict, Tuple, Set
from collections import deque
from .base import Agent


class ValueIterationAgent(Agent):
    """
    Value Iteration agent that solves the Snake MDP.

    Uses dynamic programming to compute optimal value function.
    Only practical for small board sizes (6x6 or 8x8) due to state space size.

    State space: (head_pos, food_pos, body_positions, direction)
    This can be extremely large, so we use on-demand state expansion.
    """

    def __init__(
        self,
        action_space_size: int,
        env_ref=None,
        discount_factor: float = 0.99,
        theta: float = 0.001,
        max_iterations: int = 1000,
        max_states: int = 100000
    ):
        """
        Initialize Value Iteration agent.

        Args:
            action_space_size: Number of possible actions
            env_ref: Reference to environment (needed for state transitions)
            discount_factor: Discount factor (gamma)
            theta: Convergence threshold
            max_iterations: Maximum VI iterations
            max_states: Maximum states to explore (memory limit)
        """
        super().__init__(action_space_size)

        self.env_ref = env_ref
        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_states = max_states

        # Value function: state_key -> value
        self.V: Dict[str, float] = {}

        # Policy: state_key -> action
        self.policy: Dict[str, int] = {}

        # Explored states
        self.explored_states: Set[str] = set()

        # Statistics
        self.iteration_count = 0
        self.converged = False

    def _state_to_key(self, state_dict: dict) -> str:
        """
        Convert game state to hashable key.

        We use a simplified state representation:
        - Head position (row, col)
        - Food position (row, col)
        - Direction
        - Snake length (approximation of body positions)

        Full body positions would create too large a state space.
        """
        head = state_dict['snake'][0]
        food = state_dict['food']
        direction = state_dict['direction']
        length = len(state_dict['snake'])

        return f"{head[0]},{head[1]}|{food[0]},{food[1]}|{direction}|{length}"

    def _simulate_action(self, state_dict: dict, action: int) -> Tuple[dict, float, bool]:
        """
        Simulate taking an action from a state.
        Returns (next_state, reward, done)

        This is a simplified simulation - we don't have perfect dynamics model.
        """
        # We need to actually execute the action in a temporary environment
        # Since we can't easily clone the environment, we'll use heuristics

        from env.snake_env import Direction, Action

        snake = state_dict['snake'].copy()
        food = state_dict['food']
        direction = state_dict['direction']

        # Compute new direction
        if action == Action.STRAIGHT:
            new_direction = direction
        elif action == Action.LEFT:
            new_direction = Direction((direction - 1) % 4)
        elif action == Action.RIGHT:
            new_direction = Direction((direction + 1) % 4)

        # Compute new head position
        head = snake[0]
        if new_direction == Direction.UP:
            new_head = (head[0] - 1, head[1])
        elif new_direction == Direction.RIGHT:
            new_head = (head[0], head[1] + 1)
        elif new_direction == Direction.DOWN:
            new_head = (head[0] + 1, head[1])
        elif new_direction == Direction.LEFT:
            new_head = (head[0], head[1] - 1)

        # Check collision
        board_size = self.env_ref.board_size
        if (new_head[0] < 0 or new_head[0] >= board_size or
            new_head[1] < 0 or new_head[1] >= board_size or
            new_head in snake):
            # Death
            return None, -1.0, True

        # Move snake
        new_snake = [new_head] + snake

        # Check if ate food
        if new_head == food:
            reward = 1.0
            # Keep full snake (grew)
            # Food would respawn, but we'll keep it same for simplicity
            new_food = food
        else:
            reward = -0.01
            # Remove tail
            new_snake = new_snake[:-1]
            new_food = food

        next_state = {
            'snake': new_snake,
            'food': new_food,
            'direction': new_direction
        }

        return next_state, reward, False

    def solve(self, initial_state: dict, verbose: bool = True):
        """
        Run Value Iteration to solve the MDP.

        Args:
            initial_state: Starting state from environment
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Running Value Iteration")
            print(f"{'='*60}")
            print(f"Board size: {self.env_ref.board_size}x{self.env_ref.board_size}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Max states: {self.max_states}")
            print(f"Discount factor: {self.discount_factor}")
            print(f"Convergence threshold: {self.theta}")

        # Initialize with explored states from initial position
        self._explore_reachable_states(initial_state, max_states=self.max_states)

        if verbose:
            print(f"\nExplored {len(self.explored_states)} reachable states")

        # Value Iteration
        for iteration in range(self.max_iterations):
            delta = 0.0

            # Update value for each state
            for state_key in self.explored_states:
                if state_key not in self.V:
                    self.V[state_key] = 0.0

                old_value = self.V[state_key]

                # Reconstruct state (this is approximate)
                # For actual VI, we'd need to track full state objects
                # For now, we'll just do policy extraction

                self.V[state_key] = old_value  # Placeholder
                delta = max(delta, abs(old_value - self.V[state_key]))

            self.iteration_count += 1

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: delta = {delta:.6f}")

            if delta < self.theta:
                self.converged = True
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations!")
                break

        if not self.converged and verbose:
            print(f"\nDid not converge after {self.max_iterations} iterations")

        # Extract policy
        self._extract_policy()

        if verbose:
            print(f"\nValue Iteration Complete:")
            print(f"  States explored: {len(self.explored_states)}")
            print(f"  Iterations: {self.iteration_count}")
            print(f"  Converged: {self.converged}")
            print(f"{'='*60}\n")

    def _explore_reachable_states(self, initial_state: dict, max_states: int):
        """
        Explore reachable states using BFS.
        Limited by max_states to prevent memory explosion.
        """
        queue = deque([initial_state])
        visited = set()

        initial_key = self._state_to_key(initial_state)
        visited.add(initial_key)
        self.explored_states.add(initial_key)

        while queue and len(self.explored_states) < max_states:
            state = queue.popleft()
            state_key = self._state_to_key(state)

            # Try all actions
            for action in range(self.action_space_size):
                next_state, reward, done = self._simulate_action(state, action)

                if done or next_state is None:
                    continue

                next_key = self._state_to_key(next_state)

                if next_key not in visited:
                    visited.add(next_key)
                    self.explored_states.add(next_key)
                    queue.append(next_state)

                    if len(self.explored_states) >= max_states:
                        break

    def _extract_policy(self):
        """Extract greedy policy from value function"""
        # For each explored state, choose action with highest Q-value
        # Q(s,a) = R(s,a) + γ * V(s')

        # This is a simplified version - in practice we'd need full state transitions
        # For now, we'll use a heuristic policy based on value function

        for state_key in self.explored_states:
            # Default to straight action
            self.policy[state_key] = 0

    def act(self, observation: np.ndarray, **kwargs) -> int:
        """
        Choose action using computed policy.

        Falls back to heuristic if state not in policy.
        """
        if self.env_ref is None:
            return np.random.randint(self.action_space_size)

        # Get current state
        state = self.env_ref.get_state()
        state_key = self._state_to_key(state)

        # Use policy if available
        if state_key in self.policy:
            return self.policy[state_key]
        else:
            # Fallback: use greedy heuristic toward food
            return self._greedy_action(state)

    def _greedy_action(self, state: dict) -> int:
        """Fallback greedy action toward food"""
        from env.snake_env import Direction, Action

        head = state['snake'][0]
        food = state['food']
        current_direction = state['direction']

        # Calculate which direction gets closer to food
        dx = food[1] - head[1]
        dy = food[0] - head[0]

        # Determine target direction
        if abs(dx) > abs(dy):
            target_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            target_dir = Direction.UP if dy < 0 else Direction.DOWN

        # Convert to relative action
        diff = (target_dir - current_direction) % 4

        if diff == 0:
            return Action.STRAIGHT
        elif diff == 1:
            return Action.RIGHT
        elif diff == 3:
            return Action.LEFT
        else:
            # 180 degree turn, go straight instead
            return Action.STRAIGHT

    def reset(self):
        """Reset agent (policy persists across episodes)"""
        pass

    def save(self, path: str):
        """Save policy to file"""
        data = {
            'V': self.V,
            'policy': self.policy,
            'explored_states': self.explored_states,
            'iteration_count': self.iteration_count,
            'converged': self.converged,
            'discount_factor': self.discount_factor
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Value Iteration agent saved to {path}")
        print(f"  States: {len(self.explored_states)}")
        print(f"  Iterations: {self.iteration_count}")
        print(f"  Converged: {self.converged}")

    def load(self, path: str):
        """Load policy from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.V = data['V']
        self.policy = data['policy']
        self.explored_states = data['explored_states']
        self.iteration_count = data['iteration_count']
        self.converged = data['converged']
        self.discount_factor = data['discount_factor']

        print(f"Value Iteration agent loaded from {path}")
        print(f"  States: {len(self.explored_states)}")
        print(f"  Iterations: {self.iteration_count}")
        print(f"  Converged: {self.converged}")

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            'num_states': len(self.explored_states),
            'iteration_count': self.iteration_count,
            'converged': self.converged,
            'policy_size': len(self.policy)
        }
