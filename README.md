# SnakeSense: Multi-Agent Snake Game Training Platform

A complete game environment + agent training platform where multiple agents (search, planning, RL, deep RL) learn to play Snake, with a unified evaluation harness, visualizer, and reproducible experiments.

## Project Overview

This is not just "I trained a DQN" - this is a complete ML Engineering project featuring:

- **Clean Gym-style Environment**: Reset, step, render with deterministic and randomized modes
- **Multiple Agent Families**: Heuristics, planning algorithms, RL, and deep RL
- **Reproducible Experiment Framework**: Standardized training and evaluation
- **Rich Metrics & Visualizations**: Learning curves, heatmaps, policy rollouts
- **Modern ML Engineering**: Config-driven, modular, extensible

## Architecture

```
Environment → Agents → Training Loop → Replay Buffer → Evaluator → Visualizer → Artifacts
```

### Modules

- `env/`: Core Snake game environment (Gym-style API)
- `agents/`: Pluggable agent implementations
- `rl/`: RL components (replay buffer, schedulers, training loops)
- `experiments/`: Configuration files and training/evaluation scripts
- `metrics/`: Logging and visualization utilities
- `ui/`: Pygame visualization interface

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Play Snake Manually

```bash
python play.py
```

### Train an Agent

```bash
# Train DQN agent
python experiments/train.py --config configs/dqn.yaml

# Train Q-Learning agent
python experiments/train.py --config configs/qlearning.yaml
```

### Evaluate Agents

```bash
python experiments/evaluate.py --agent dqn --checkpoint checkpoints/dqn_best.pt
```

### Generate Visualizations

```bash
python metrics/plot_results.py --results reports/experiment_results.json
```

## Agent Implementations

### Baseline Agents
- **Random Agent**: Random action selection
- **Greedy Agent**: Shortest path to food
- **A* Agent**: Heuristic path planning with obstacle awareness

### Planning Agents
- **Value Iteration**: Tabular MDP solver (small boards)
- **Policy Evaluation**: For reporting and analysis

### Reinforcement Learning Agents
- **Q-Learning**: Tabular RL with epsilon-greedy exploration
- **DQN**: Deep Q-Network with experience replay
- **Double DQN**: Reduced overestimation bias (optional)
- **Dueling DQN**: Separate value and advantage streams (optional)

## State Representations

### Feature Vector (Tabular-Friendly)
- Direction to food (dx, dy)
- Danger indicators (straight/left/right)
- Current direction (one-hot)
- Distance to walls
- Snake length

Used by: Q-Learning, Value Iteration

### Grid Tensor (Deep RL-Friendly)
- 3 channels: snake head, snake body, food
- Used by: DQN (MLP or CNN)

## Action Space

3 actions relative to current heading:
- Straight
- Left
- Right

(Prevents immediate death from 180° turns)

## Reward Structure

- `+1.0`: Eating food
- `-1.0`: Death (collision)
- `-0.01`: Small step penalty (encourages efficiency)
- Optional: Dense shaping based on distance to food (ablation study)

## Metrics Tracked

- Average score over last 100 episodes
- Max score achieved
- Mean survival steps
- Food per 1,000 steps
- Success rate (score > threshold)
- Training wall-clock time

## Experiment Results

| Agent | Avg Score | Max Score | Survival Time | Food Rate | Training Time |
|-------|-----------|-----------|---------------|-----------|---------------|
| Random | TBD | TBD | TBD | TBD | - |
| Greedy | TBD | TBD | TBD | TBD | - |
| A* | TBD | TBD | TBD | TBD | - |
| Q-Learning | TBD | TBD | TBD | TBD | TBD |
| DQN | TBD | TBD | TBD | TBD | TBD |

## Project Structure

```
SnakeSense/
├── env/                    # Snake environment
│   ├── __init__.py
│   ├── snake_env.py       # Main environment class
│   └── utils.py           # Helper functions
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── base.py            # Base agent interface
│   ├── random_agent.py
│   ├── greedy_agent.py
│   ├── astar_agent.py
│   ├── value_iteration.py
│   ├── qlearning_agent.py
│   └── dqn_agent.py
├── rl/                     # RL components
│   ├── __init__.py
│   ├── replay_buffer.py
│   ├── schedulers.py
│   ├── rewards.py
│   └── training.py
├── experiments/            # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── metrics/                # Logging and visualization
│   ├── __init__.py
│   ├── logger.py
│   └── plot_results.py
├── ui/                     # Visualization
│   ├── __init__.py
│   └── renderer.py
├── configs/                # Configuration files
│   ├── default.yaml
│   ├── dqn.yaml
│   └── qlearning.yaml
├── reports/                # Generated reports and plots
├── checkpoints/            # Model checkpoints
├── play.py                 # Manual play script
├── requirements.txt
├── Makefile
└── README.md
```

## Design Choices & Tradeoffs

### Relative vs Absolute Actions
- **Choice**: 3 relative actions (straight/left/right) instead of 4 absolute directions
- **Rationale**: Prevents immediate death from 180° turns, simplifies learning

### State Representation
- **Choice**: Support both feature vectors and grid tensors
- **Rationale**: Enables comparison between tabular and deep RL methods

### Reward Shaping
- **Choice**: Small negative step penalty
- **Rationale**: Encourages efficient food collection without excessive survival bias

## Development Roadmap

- [x] Phase 0: Repo + skeleton
- [ ] Phase 1: Environment + renderer
- [ ] Phase 2: Baseline agents
- [ ] Phase 3: Tabular planning + RL
- [ ] Phase 4: DQN
- [ ] Phase 5: Evaluation harness + visualization
- [ ] Phase 6: Polish + documentation

## Future Extensions

- Reward shaping ablation studies
- Curriculum learning (small → large boards)
- Prioritized experience replay
- CNN state encoder
- LLM-generated experiment analysis
- Web demo interface

## Contributing

This is a research/educational project. Feel free to extend it with your own agents and experiments!

## License

MIT License
