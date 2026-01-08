# Getting Started with SnakeSense

This guide will help you set up and run SnakeSense on your machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SnakeSense.git
cd SnakeSense
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- pygame (visualization)
- torch (deep learning)
- matplotlib & seaborn (plotting)
- pyyaml (configuration)
- tqdm (progress bars)
- and more...

### 4. Verify Installation

```bash
python test_environment.py
```

You should see:
```
============================================================
SNAKESENSE - Environment & Agent Tests
============================================================
Testing Snake Environment...
  ✓ Environment test passed!
Testing Agents...
  ✓ All agents work!
...
✓ ALL TESTS PASSED!
============================================================
```

## Quick Start Examples

### 1. Play Snake Yourself

```bash
python play.py
```

**Controls:**
- Arrow keys (or WASD): Move snake (relative turns)
- SPACE: Restart game
- ESC: Quit

### 2. Watch AI Agents Play

**Watch Greedy Agent:**
```bash
python play.py --mode agent --agent greedy --games 5
```

**Watch A* Agent:**
```bash
python play.py --mode agent --agent astar --games 5
```

**Watch Random Agent (for comparison):**
```bash
python play.py --mode agent --agent random --games 3
```

### 3. Evaluate Baseline Agents

Run benchmark on baseline agents:

```bash
python experiments/evaluate.py --episodes 100
```

This will test Random, Greedy, and A* agents and save results to [reports/baseline_results.json](reports/baseline_results.json).

**Plot the results:**
```bash
python metrics/plot_results.py --baseline reports/baseline_results.json
```

This creates [reports/baseline_comparison.png](reports/baseline_comparison.png).

### 4. Train Q-Learning Agent

Train a tabular Q-Learning agent:

```bash
python experiments/train.py --config configs/qlearning.yaml
```

Training will take a few minutes. Progress is shown with:
- Episode rewards
- Q-table size
- Epsilon decay
- Evaluation scores every 100 episodes

**Checkpoints saved to:** [checkpoints/qlearning_final.pkl](checkpoints/)

**Metrics saved to:** [checkpoints/qlearning_metrics.json](checkpoints/)

### 5. Train DQN Agent

Train a Deep Q-Network:

```bash
python experiments/train.py --config configs/dqn.yaml
```

This will:
- Initialize neural network
- Fill replay buffer
- Train with experience replay
- Update target network periodically

**Training takes longer** (10,000 episodes) but achieves better performance!

**Checkpoints saved to:** [checkpoints/dqn_final.pt](checkpoints/)

### 6. Visualize Training Results

**Plot Q-Learning curves:**
```bash
python metrics/plot_results.py --metrics checkpoints/qlearning_metrics.json
```

**Plot DQN curves:**
```bash
python metrics/plot_results.py --metrics checkpoints/dqn_metrics.json
```

**Compare multiple agents:**
```bash
python metrics/plot_results.py --metrics checkpoints/qlearning_metrics.json checkpoints/dqn_metrics.json --compare
```

## Project Workflow

Here's the typical workflow:

```
1. Test environment        → python test_environment.py
2. Play manually          → python play.py
3. Watch baseline agents  → python play.py --mode agent
4. Evaluate baselines     → python experiments/evaluate.py
5. Train Q-Learning       → python experiments/train.py --config configs/qlearning.yaml
6. Train DQN              → python experiments/train.py --config configs/dqn.yaml
7. Plot results           → python metrics/plot_results.py --compare --metrics ...
8. Analyze & iterate!
```

## Configuration

All configurations are in [configs/](configs/) as YAML files:

- `default.yaml` - Base configuration
- `random.yaml` - Random agent baseline
- `qlearning.yaml` - Q-Learning hyperparameters
- `dqn.yaml` - DQN hyperparameters

### Key Parameters

**Environment:**
- `board_size`: Grid size (default: 10x10)
- `state_type`: 'feature' (14-dim vector) or 'grid' (3-channel tensor)
- `deterministic`: Fixed or random food spawning

**Rewards:**
- `food`: +1.0 (eating food)
- `death`: -1.0 (collision)
- `step`: -0.01 (small penalty per step)

**Training:**
- `episodes`: Number of training episodes
- `learning_rate`: Step size for updates
- `epsilon_decay`: Exploration decay rate
- `batch_size`: (DQN only) Mini-batch size
- `memory_size`: (DQN only) Replay buffer capacity

## Troubleshooting

### ModuleNotFoundError

Make sure you've installed requirements:
```bash
pip install -r requirements.txt
```

### Pygame Window Issues

If pygame window doesn't appear:
- On macOS: May need XQuartz for remote sessions
- Try running locally instead of over SSH
- Check pygame installation: `python -c "import pygame; print(pygame.version.ver)"`

### CUDA/GPU Issues

DQN will automatically use GPU if available. To force CPU:
```python
# In configs/dqn.yaml, you can't directly set device, but it auto-detects
# To force CPU, set environment variable:
export CUDA_VISIBLE_DEVICES=""
```

### Training Takes Too Long

For faster experimentation:
- Reduce board size (8x8 instead of 10x10)
- Reduce episodes (1000 instead of 5000)
- Use smaller networks (hidden_dims: [64, 64])

## Next Steps

1. **Experiment with reward shaping** - Modify rewards in config files
2. **Try different board sizes** - Compare 8x8 vs 10x10 vs 12x12
3. **Tune hyperparameters** - Learning rate, epsilon decay, network size
4. **Implement improvements** - Double DQN, Dueling DQN, Prioritized Replay
5. **Add curriculum learning** - Train on small boards then scale up

## Getting Help

- Check [README.md](README.md) for architecture details
- Review config files in [configs/](configs/)
- Read agent implementations in [agents/](agents/)
- Post issues on GitHub

Enjoy building your Snake AI! 🐍🤖
