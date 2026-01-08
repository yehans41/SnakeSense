#!/usr/bin/env python3
"""
Quick test script to verify environment and agents work correctly
"""

import sys
from env import SnakeEnv
from agents import RandomAgent, GreedyAgent, AStarAgent, ValueIterationAgent

def test_environment():
    """Test basic environment functionality"""
    print("Testing Snake Environment...")

    env = SnakeEnv(board_size=10, state_type='feature')
    obs = env.reset()

    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space size: {env.observation_space_size}")
    print(f"  Action space size: {env.action_space_size}")

    # Take a few steps
    for i in range(5):
        action = env.action_space_size - 1  # Turn right
        obs, reward, done, info = env.step(action)

        if done:
            print(f"  Episode ended after {i+1} steps")
            break

    print("  ✓ Environment test passed!")
    return True

def test_agents():
    """Test that all agents can act"""
    print("\nTesting Agents...")

    env = SnakeEnv(board_size=10, state_type='feature')
    obs = env.reset()

    # Test Random Agent
    agent = RandomAgent(action_space_size=3)
    action = agent.act(obs)
    print(f"  Random agent action: {action}")
    print("  ✓ Random agent works!")

    # Test Greedy Agent
    agent = GreedyAgent(action_space_size=3, env_ref=env)
    action = agent.act(obs)
    print(f"  Greedy agent action: {action}")
    print("  ✓ Greedy agent works!")

    # Test A* Agent
    agent = AStarAgent(action_space_size=3, env_ref=env)
    action = agent.act(obs)
    print(f"  A* agent action: {action}")
    print("  ✓ A* agent works!")

    # Test Value Iteration Agent
    agent = ValueIterationAgent(action_space_size=3, env_ref=env)
    action = agent.act(obs)
    print(f"  Value Iteration agent action: {action}")
    print("  ✓ Value Iteration agent works!")

    return True

def test_episode():
    """Test running a complete episode"""
    print("\nTesting Complete Episode...")

    env = SnakeEnv(board_size=10, state_type='feature')
    agent = GreedyAgent(action_space_size=3, env_ref=env)

    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

    print(f"  Episode completed:")
    print(f"    Steps: {info['steps']}")
    print(f"    Score: {info['score']}")
    print(f"    Final snake length: {len(env.get_state()['snake'])}")
    print("  ✓ Episode test passed!")

    return True

def main():
    print("=" * 60)
    print("SNAKESENSE - Environment & Agent Tests")
    print("=" * 60)

    try:
        test_environment()
        test_agents()
        test_episode()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now:")
        print("  1. Play manually: python play.py")
        print("  2. Watch agents: python play.py --mode agent --agent greedy")
        print("  3. Evaluate baselines: python experiments/evaluate.py")
        print("  4. Train RL agents: python experiments/train.py --config configs/qlearning.yaml")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
