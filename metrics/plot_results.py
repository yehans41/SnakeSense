#!/usr/bin/env python3
"""
Visualization and plotting utilities for training results
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def smooth(data, weight=0.9):
    """Exponential moving average smoothing"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_curves(metrics_path, output_dir):
    """Plot training curves from metrics file"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    agent_name = Path(metrics_path).stem.replace('_metrics', '')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{agent_name.upper()} Training Results', fontsize=16)

    # 1. Episode Rewards
    ax = axes[0, 0]
    rewards = metrics['episode_rewards']
    episodes = list(range(1, len(rewards) + 1))

    ax.plot(episodes, rewards, alpha=0.3, label='Raw', color='blue')
    ax.plot(episodes, smooth(rewards, 0.9), label='Smoothed', color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Episode Scores
    ax = axes[0, 1]
    scores = metrics['episode_scores']

    ax.plot(episodes, scores, alpha=0.3, label='Raw', color='green')
    ax.plot(episodes, smooth(scores, 0.9), label='Smoothed', color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Episode Scores (Food Collected)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Episode Steps (Survival Time)
    ax = axes[1, 0]
    steps = metrics['episode_steps']

    ax.plot(episodes, steps, alpha=0.3, label='Raw', color='orange')
    ax.plot(episodes, smooth(steps, 0.9), label='Smoothed', color='orange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Survival Steps per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Evaluation Scores
    ax = axes[1, 1]
    if metrics['eval_scores']:
        eval_episodes, eval_scores = zip(*metrics['eval_scores'])
        ax.plot(eval_episodes, eval_scores, marker='o', color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Score')
        ax.set_title('Evaluation Performance (10 episodes, greedy)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
        ax.set_title('Evaluation Performance')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'{agent_name}_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")

    plt.close()

    # Additional plot for DQN: Loss curve
    if 'losses' in metrics and metrics['losses']:
        fig, ax = plt.subplots(figsize=(12, 6))

        losses = metrics['losses']
        episodes_with_loss = list(range(1, len(losses) + 1))

        ax.plot(episodes_with_loss, losses, alpha=0.3, label='Raw', color='purple')
        ax.plot(episodes_with_loss, smooth(losses, 0.95), label='Smoothed', color='purple', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title(f'{agent_name.upper()} - Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / f'{agent_name}_loss_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to: {output_path}")

        plt.close()


def plot_comparison(metrics_files, output_dir):
    """Plot comparison between multiple agents"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    all_metrics = {}
    for metrics_path in metrics_files:
        agent_name = Path(metrics_path).stem.replace('_metrics', '')
        with open(metrics_path, 'r') as f:
            all_metrics[agent_name] = json.load(f)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Comparison', fontsize=16)

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

    # 1. Scores comparison
    ax = axes[0, 0]
    for i, (agent_name, metrics) in enumerate(all_metrics.items()):
        scores = metrics['episode_scores']
        episodes = list(range(1, len(scores) + 1))
        smoothed = smooth(scores, 0.9)
        ax.plot(episodes, smoothed, label=agent_name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Scores Comparison (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Rewards comparison
    ax = axes[0, 1]
    for i, (agent_name, metrics) in enumerate(all_metrics.items()):
        rewards = metrics['episode_rewards']
        episodes = list(range(1, len(rewards) + 1))
        smoothed = smooth(rewards, 0.9)
        ax.plot(episodes, smoothed, label=agent_name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards Comparison (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Survival steps comparison
    ax = axes[1, 0]
    for i, (agent_name, metrics) in enumerate(all_metrics.items()):
        steps = metrics['episode_steps']
        episodes = list(range(1, len(steps) + 1))
        smoothed = smooth(steps, 0.9)
        ax.plot(episodes, smoothed, label=agent_name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Survival Steps Comparison (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Final performance bar chart
    ax = axes[1, 1]
    agent_names = list(all_metrics.keys())
    final_scores = [np.mean(m['episode_scores'][-100:]) for m in all_metrics.values()]

    bars = ax.bar(agent_names, final_scores, color=colors[:len(agent_names)])
    ax.set_ylabel('Average Score (last 100 episodes)')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.tight_layout()

    output_path = output_dir / 'agent_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    plt.close()


def plot_baseline_comparison(baseline_results_path, output_dir):
    """Plot comparison of baseline agents"""
    with open(baseline_results_path, 'r') as f:
        results = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Baseline Agents Comparison', fontsize=16)

    agents = list(results.keys())
    colors_map = {'random': 'red', 'greedy': 'blue', 'astar': 'green'}

    # 1. Average Scores
    ax = axes[0, 0]
    scores = [results[a]['avg_score'] for a in agents]
    stds = [results[a]['std_score'] for a in agents]
    bars = ax.bar(agents, scores, yerr=stds, capsize=5,
                  color=[colors_map.get(a, 'gray') for a in agents])
    ax.set_ylabel('Average Score')
    ax.set_title('Average Score ± Std')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Max Scores
    ax = axes[0, 1]
    max_scores = [results[a]['max_score'] for a in agents]
    ax.bar(agents, max_scores, color=[colors_map.get(a, 'gray') for a in agents])
    ax.set_ylabel('Max Score')
    ax.set_title('Maximum Score Achieved')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Average Steps
    ax = axes[1, 0]
    avg_steps = [results[a]['avg_steps'] for a in agents]
    ax.bar(agents, avg_steps, color=[colors_map.get(a, 'gray') for a in agents])
    ax.set_ylabel('Average Steps')
    ax.set_title('Average Survival Steps')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Food per 1000 steps
    ax = axes[1, 1]
    food_rate = [results[a]['food_per_1000_steps'] for a in agents]
    ax.bar(agents, food_rate, color=[colors_map.get(a, 'gray') for a in agents])
    ax.set_ylabel('Food per 1000 steps')
    ax.set_title('Food Collection Rate')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'baseline_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Baseline comparison saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='Path(s) to metrics JSON file(s)'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        help='Path to baseline results JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Create comparison plot (requires multiple metrics files)'
    )

    args = parser.parse_args()

    if not args.metrics and not args.baseline:
        print("Error: Must provide either --metrics or --baseline")
        sys.exit(1)

    # Plot baseline results if provided
    if args.baseline:
        print(f"Plotting baseline results from {args.baseline}...")
        plot_baseline_comparison(args.baseline, args.output_dir)

    # Plot training curves
    if args.metrics:
        if args.compare and len(args.metrics) > 1:
            print("Plotting comparison...")
            plot_comparison(args.metrics, args.output_dir)
        else:
            for metrics_path in args.metrics:
                print(f"Plotting training curves from {metrics_path}...")
                plot_training_curves(metrics_path, args.output_dir)

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
