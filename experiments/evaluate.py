#!/usr/bin/env python3
"""
Evaluation script for Snake agents.
Tests agents and generates performance metrics.
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import SnakeEnv
from agents import RandomAgent, GreedyAgent, AStarAgent, ValueIterationAgent


def evaluate_agent(agent, env, num_episodes: int = 100, verbose: bool = True):
    """
    Evaluate an agent over multiple episodes.

    Args:
        agent: Agent to evaluate
        env: Environment instance
        num_episodes: Number of episodes to run
        verbose: Print progress

    Returns:
        Dictionary of metrics
    """
    scores = []
    steps_list = []
    food_collected = []

    for episode in range(num_episodes):
        obs = env.reset()
        agent.reset()
        done = False
        episode_steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episode_steps += 1

        scores.append(info['score'])
        steps_list.append(episode_steps)
        food_collected.append(info['food_collected'])

        if verbose and (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}: Score={info['score']}, Steps={episode_steps}")

    # Calculate metrics
    scores = np.array(scores)
    steps_list = np.array(steps_list)
    food_collected = np.array(food_collected)

    metrics = {
        'avg_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
        'avg_steps': float(np.mean(steps_list)),
        'std_steps': float(np.std(steps_list)),
        'max_steps': int(np.max(steps_list)),
        'avg_food_collected': float(np.mean(food_collected)),
        'food_per_1000_steps': float(np.sum(food_collected) / np.sum(steps_list) * 1000),
        'success_rate': float(np.mean(scores > 0)),
        'num_episodes': num_episodes
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Snake agents")
    parser.add_argument(
        '--agents',
        nargs='+',
        default=['random', 'greedy', 'astar'],
        choices=['random', 'greedy', 'astar', 'vi'],
        help='Agents to evaluate'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes per agent'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        default=10,
        help='Board size'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/baseline_results.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SNAKESENSE - Baseline Agent Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Board size: {args.board_size}x{args.board_size}")
    print(f"  Episodes per agent: {args.episodes}")
    print(f"  Random seed: {args.seed}")
    print(f"  Agents: {', '.join(args.agents)}")
    print()

    results = {}

    for agent_name in args.agents:
        print(f"\nEvaluating {agent_name.upper()} agent...")
        print("-" * 70)

        # Create environment
        env = SnakeEnv(
            board_size=args.board_size,
            state_type='feature',
            deterministic=False,
            seed=args.seed
        )

        # Create agent
        if agent_name == 'random':
            agent = RandomAgent(action_space_size=3, seed=args.seed)
        elif agent_name == 'greedy':
            agent = GreedyAgent(action_space_size=3, env_ref=env)
        elif agent_name == 'astar':
            agent = AStarAgent(action_space_size=3, env_ref=env)
        elif agent_name == 'vi':
            agent = ValueIterationAgent(action_space_size=3, env_ref=env)
            print("  Note: VI agent uses greedy heuristic (full VI solve not performed)")
        else:
            print(f"Unknown agent: {agent_name}")
            continue

        # Evaluate
        metrics = evaluate_agent(agent, env, num_episodes=args.episodes, verbose=True)

        results[agent_name] = metrics

        # Print summary
        print(f"\n{agent_name.upper()} Results:")
        print(f"  Average Score: {metrics['avg_score']:.2f} ± {metrics['std_score']:.2f}")
        print(f"  Max Score: {metrics['max_score']}")
        print(f"  Average Steps: {metrics['avg_steps']:.2f} ± {metrics['std_steps']:.2f}")
        print(f"  Food per 1000 steps: {metrics['food_per_1000_steps']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {args.output}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Agent':<12} {'Avg Score':<12} {'Max Score':<12} {'Avg Steps':<12} {'Food/1k':<12}")
    print("-" * 70)

    for agent_name in args.agents:
        if agent_name in results:
            m = results[agent_name]
            print(f"{agent_name:<12} {m['avg_score']:>10.2f}  {m['max_score']:>10}  "
                  f"{m['avg_steps']:>10.1f}  {m['food_per_1000_steps']:>10.2f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
