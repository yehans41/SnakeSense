#!/usr/bin/env python3
"""
Training script for RL agents (Q-Learning and DQN)
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from env import SnakeEnv
from agents import QLearningAgent, DQNAgent


def train_qlearning(env, agent, config, save_dir):
    """Train Q-Learning agent"""
    episodes = config['training']['episodes']
    max_steps = config['training']['max_steps']
    eval_frequency = config['training']['eval_frequency']
    save_frequency = config['training']['save_frequency']

    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_scores = []
    eval_scores = []

    print(f"\nTraining Q-Learning for {episodes} episodes...")
    print(f"Initial epsilon: {agent.epsilon:.4f}")

    start_time = time.time()

    for episode in tqdm(range(episodes), desc="Training"):
        obs = env.reset()
        agent.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            # Select action
            action = agent.act(obs)

            # Take step
            next_obs, reward, done, info = env.step(action)

            # Update agent
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            steps += 1

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_scores.append(info['score'])

        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_score = evaluate_agent(agent, env, num_episodes=10, epsilon=0.0)
            eval_scores.append((episode + 1, eval_score))

            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Avg Score (last 100): {np.mean(episode_scores[-100:]):.2f}")
            print(f"  Eval Score: {eval_score:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table size: {len(agent.q_table)}")

        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = save_dir / f"qlearning_ep{episode+1}.pkl"
            agent.save(str(checkpoint_path))

    # Save final model
    final_path = save_dir / "qlearning_final.pkl"
    agent.save(str(final_path))

    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'episode_scores': episode_scores,
        'eval_scores': eval_scores,
        'training_time': time.time() - start_time,
        'config': config
    }

    metrics_path = save_dir / "qlearning_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics


def train_dqn(env, agent, config, save_dir):
    """Train DQN agent"""
    episodes = config['training']['episodes']
    max_steps = config['training']['max_steps']
    eval_frequency = config['training']['eval_frequency']
    save_frequency = config['training']['save_frequency']
    warmup_episodes = config['training'].get('warmup_episodes', 100)

    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_scores = []
    eval_scores = []
    losses = []

    print(f"\nTraining DQN for {episodes} episodes...")
    print(f"Warmup episodes: {warmup_episodes}")
    print(f"Initial epsilon: {agent.epsilon:.4f}")
    print(f"Device: {agent.device}")

    start_time = time.time()

    for episode in tqdm(range(episodes), desc="Training"):
        obs = env.reset()
        agent.reset()
        done = False
        episode_reward = 0
        steps = 0

        # Use epsilon=1.0 during warmup for pure exploration
        epsilon = 1.0 if episode < warmup_episodes else agent.epsilon

        while not done and steps < max_steps:
            # Select action
            action = agent.act(obs, epsilon=epsilon)

            # Take step
            next_obs, reward, done, info = env.step(action)

            # Store experience and train
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            steps += 1

        # Decay epsilon (only after warmup)
        if episode >= warmup_episodes:
            agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_scores.append(info['score'])

        if agent.losses:
            losses.append(np.mean(agent.losses[-steps:]))

        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_score = evaluate_agent(agent, env, num_episodes=10, epsilon=0.0)
            eval_scores.append((episode + 1, eval_score))

            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Avg Score (last 100): {np.mean(episode_scores[-100:]):.2f}")
            print(f"  Eval Score: {eval_score:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Memory size: {len(agent.memory)}")
            if losses:
                print(f"  Avg Loss: {losses[-1]:.4f}")

        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = save_dir / f"dqn_ep{episode+1}.pt"
            agent.save(str(checkpoint_path))

    # Save final model
    final_path = save_dir / "dqn_final.pt"
    agent.save(str(final_path))

    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'episode_scores': episode_scores,
        'eval_scores': eval_scores,
        'losses': losses,
        'training_time': time.time() - start_time,
        'config': config
    }

    metrics_path = save_dir / "dqn_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics


def evaluate_agent(agent, env, num_episodes=10, epsilon=0.0):
    """Evaluate agent performance"""
    scores = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs, epsilon=epsilon)
            obs, reward, done, info = env.step(action)

        scores.append(info['score'])

    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for Snake")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints',
        help='Output directory for checkpoints'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = SnakeEnv(
        board_size=config['environment']['board_size'],
        state_type=config['environment']['state_type'],
        deterministic=config['environment']['deterministic'],
        seed=config['environment'].get('seed'),
        reward_config=config.get('rewards')
    )

    print("=" * 70)
    print("SNAKESENSE - Agent Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Agent type: {config['agent']['type']}")
    print(f"  Board size: {env.board_size}x{env.board_size}")
    print(f"  State type: {env.state_type}")
    print(f"  Episodes: {config['training']['episodes']}")

    # Create and train agent
    agent_type = config['agent']['type']

    if agent_type == 'qlearning':
        agent = QLearningAgent(
            action_space_size=env.action_space_size,
            learning_rate=config['agent']['learning_rate'],
            discount_factor=config['agent']['discount_factor'],
            epsilon_start=config['agent']['epsilon_start'],
            epsilon_end=config['agent']['epsilon_end'],
            epsilon_decay=config['agent']['epsilon_decay'],
            seed=config['agent'].get('seed')
        )
        train_qlearning(env, agent, config, save_dir)

    elif agent_type == 'dqn':
        agent = DQNAgent(
            state_dim=env.observation_space_size,
            action_space_size=env.action_space_size,
            learning_rate=config['agent']['learning_rate'],
            discount_factor=config['agent']['discount_factor'],
            epsilon_start=config['agent']['epsilon_start'],
            epsilon_end=config['agent']['epsilon_end'],
            epsilon_decay=config['agent']['epsilon_decay'],
            batch_size=config['agent']['batch_size'],
            memory_size=config['agent']['memory_size'],
            target_update_frequency=config['agent']['target_update_frequency'],
            hidden_dims=config['agent']['hidden_dims'],
            seed=config['agent'].get('seed')
        )
        train_dqn(env, agent, config, save_dir)

    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)


if __name__ == '__main__':
    main()
