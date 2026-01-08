# SnakeSense Command Reference

Complete guide to all available commands and functionality.

## 🧪 Testing Commands

### Basic Environment Test
```bash
python3 test_environment.py
```
**Tests:** Environment, all agents, episode execution
**Duration:** ~5 seconds
**Output:** Pass/fail for each component

### Comprehensive Functionality Test
```bash
python3 test_all_functionality.py
```
**Tests:** All 10 core components including:
- Module imports
- Environment creation (feature & grid)
- State representations
- All 7 agent types
- Episode execution
- Replay buffer
- Q-Learning updates
- DQN training
- Config files
- Save/load functionality

**Duration:** ~30 seconds
**Output:** Detailed test results for each component

---

## 🎮 Interactive Play Commands

### Play Snake Yourself
```bash
python3 play.py
```
**Controls:**
- `W` or `↑`: Continue straight
- `A` or `←`: Turn left
- `D` or `→`: Turn right
- `SPACE`: Restart game
- `ESC`: Quit

**Options:**
```bash
python3 play.py --board-size 12 --fps 15
```

### Watch AI Agents Play

**Watch Random Agent:**
```bash
python3 play.py --mode agent --agent random --games 3
```

**Watch Greedy Agent:**
```bash
python3 play.py --mode agent --agent greedy --games 5 --fps 10
```

**Watch A* Agent:**
```bash
python3 play.py --mode agent --agent astar --games 3 --fps 8
```

**Watch Value Iteration:**
```bash
python3 play.py --mode agent --agent vi --board-size 6 --games 3
```

**All Options:**
```bash
python3 play.py \
  --mode agent \
  --agent greedy \
  --board-size 10 \
  --fps 10 \
  --games 5
```

**Parameters:**
- `--mode`: `human` or `agent` (default: human)
- `--agent`: `random`, `greedy`, `astar`, `vi` (default: greedy)
- `--board-size`: Grid size (default: 10)
- `--fps`: Frames per second (default: 10)
- `--games`: Number of games to play (default: 1)

---

## 📊 Evaluation Commands

### Evaluate All Baseline Agents
```bash
python3 experiments/evaluate.py --episodes 100
```
**Output:** [reports/baseline_results.json](reports/baseline_results.json)
**Duration:** ~2-5 minutes
**Agents tested:** Random, Greedy, A*

### Evaluate Specific Agents
```bash
python3 experiments/evaluate.py --agents greedy astar --episodes 50
```

### Evaluate with Different Settings
```bash
python3 experiments/evaluate.py \
  --agents random greedy astar vi \
  --episodes 100 \
  --board-size 8 \
  --seed 42 \
  --output reports/custom_results.json
```

**Parameters:**
- `--agents`: Space-separated list (`random`, `greedy`, `astar`, `vi`)
- `--episodes`: Number of episodes per agent (default: 100)
- `--board-size`: Grid size (default: 10)
- `--seed`: Random seed (default: 42)
- `--output`: Output JSON file (default: reports/baseline_results.json)

---

## 🤖 Training Commands

### Train Q-Learning Agent

**Standard Training (5000 episodes):**
```bash
python3 experiments/train.py --config configs/qlearning.yaml
```
**Duration:** ~10-15 minutes
**Output:**
- Checkpoints: [checkpoints/qlearning_final.pkl](checkpoints/)
- Metrics: [checkpoints/qlearning_metrics.json](checkpoints/)

**Quick Demo (500 episodes):**
```bash
python3 experiments/train.py --config configs/qlearning_quick.yaml
```
**Duration:** ~2-3 minutes

**Custom Configuration:**
```bash
python3 experiments/train.py \
  --config configs/qlearning.yaml \
  --output-dir checkpoints_custom
```

### Train DQN Agent

**Standard Training (10,000 episodes):**
```bash
python3 experiments/train.py --config configs/dqn.yaml
```
**Duration:** ~30-60 minutes (uses PyTorch, GPU if available)
**Output:**
- Checkpoints: [checkpoints/dqn_final.pt](checkpoints/)
- Metrics: [checkpoints/dqn_metrics.json](checkpoints/)

**Quick Demo (1000 episodes):**
Create custom config with fewer episodes:
```yaml
# configs/dqn_quick.yaml
training:
  episodes: 1000
```

```bash
python3 experiments/train.py --config configs/dqn_quick.yaml
```

**Parameters:**
- `--config`: Path to YAML config file (required)
- `--output-dir`: Checkpoint directory (default: checkpoints)

---

## 📈 Visualization Commands

### Plot Baseline Results
```bash
python3 metrics/plot_results.py \
  --baseline reports/baseline_results.json \
  --output-dir reports
```
**Output:** [reports/baseline_comparison.png](reports/baseline_comparison.png)
**Shows:** 4 comparison plots (scores, steps, efficiency, max scores)

### Plot Training Curves

**Single Agent:**
```bash
python3 metrics/plot_results.py \
  --metrics checkpoints/qlearning_metrics.json \
  --output-dir reports
```
**Output:**
- [reports/qlearning_training_curves.png](reports/)
- Shows rewards, scores, steps, eval performance

**Multiple Agents (Comparison):**
```bash
python3 metrics/plot_results.py \
  --metrics checkpoints/qlearning_metrics.json checkpoints/dqn_metrics.json \
  --compare \
  --output-dir reports
```
**Output:** [reports/agent_comparison.png](reports/)
**Shows:** Side-by-side comparison of training progress

**All Options:**
```bash
python3 metrics/plot_results.py \
  --metrics checkpoints/qlearning_metrics.json \
  --baseline reports/baseline_results.json \
  --output-dir reports
```

**Parameters:**
- `--metrics`: Path(s) to metrics JSON file(s)
- `--baseline`: Path to baseline results JSON
- `--compare`: Create comparison plot (multi-agent)
- `--output-dir`: Output directory (default: reports)

---

## 🛠️ Utility Commands

### Quick Demo Script
```bash
./run_demo.sh
```
**Runs:**
1. Environment tests
2. Baseline evaluation (100 episodes)
3. Plot generation

**Duration:** ~5 minutes

### Make Commands
```bash
# Install dependencies
make install

# Play manually
make play

# Evaluate baselines
make baseline

# Train Q-Learning
make train-qlearning

# Train DQN
make train-dqn

# Generate plots
make plot

# Clean outputs
make clean
```

---

## 🔧 Configuration Files

All configs in [configs/](configs/):

### `default.yaml`
Base configuration template

### `random.yaml`
Random agent baseline evaluation

### `qlearning.yaml`
Q-Learning training (5000 episodes, 8x8 board)

### `qlearning_quick.yaml`
Quick Q-Learning demo (500 episodes)

### `dqn.yaml`
DQN training (10,000 episodes, 10x10 board)

### `value_iteration.yaml`
Value Iteration (small 6x6 board, deterministic)

---

## 📁 File Structure Reference

```
SnakeSense/
├── test_environment.py           # Quick test
├── test_all_functionality.py     # Comprehensive test
├── play.py                        # Interactive play
├── run_demo.sh                    # Quick demo
│
├── experiments/
│   ├── train.py                   # Training script
│   └── evaluate.py                # Evaluation script
│
├── metrics/
│   └── plot_results.py            # Visualization
│
├── configs/                       # Configuration files
├── checkpoints/                   # Saved models
└── reports/                       # Results & plots
```

---

## 🎯 Common Workflows

### 1. Quick Verification
```bash
python3 test_all_functionality.py
```

### 2. Baseline Evaluation
```bash
python3 experiments/evaluate.py --episodes 100
python3 metrics/plot_results.py --baseline reports/baseline_results.json
```

### 3. Train & Visualize Q-Learning
```bash
python3 experiments/train.py --config configs/qlearning_quick.yaml
python3 metrics/plot_results.py --metrics checkpoints/qlearning_metrics.json
```

### 4. Train & Compare Multiple Agents
```bash
# Train both
python3 experiments/train.py --config configs/qlearning.yaml
python3 experiments/train.py --config configs/dqn.yaml

# Compare
python3 metrics/plot_results.py \
  --metrics checkpoints/qlearning_metrics.json checkpoints/dqn_metrics.json \
  --compare
```

### 5. Watch Best Agent
```bash
# First evaluate to find best
python3 experiments/evaluate.py --episodes 100

# Then watch it play
python3 play.py --mode agent --agent greedy --games 5 --fps 15
```

---

## 🔍 Troubleshooting

### Import Errors
```bash
pip3 install -r requirements.txt
```

### Pygame Display Issues
Run locally (not over SSH), or use headless mode:
```bash
export SDL_VIDEODRIVER=dummy
python3 experiments/evaluate.py  # No GUI needed
```

### CUDA/GPU Issues
Force CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""
python3 experiments/train.py --config configs/dqn.yaml
```

### Check Training Progress
```bash
# If training in background
cat checkpoints/qlearning_metrics.json | python3 -m json.tool | tail -20
```

---

## 📊 Expected Output Examples

### Baseline Evaluation Results
```
Random:  Avg=0.12, Max=1,  Steps=18.5
Greedy:  Avg=17.3, Max=32, Steps=140.8
A*:      Avg=14.4, Max=39, Steps=109.7
```

### Q-Learning Training Progress
```
Episode 500/5000
  Avg Reward (last 100): 3.45
  Avg Score (last 100): 8.23
  Eval Score: 12.5
  Epsilon: 0.606
  Q-table size: 8,432
```

### DQN Training Progress
```
Episode 1000/10000
  Avg Reward (last 100): 5.67
  Avg Score (last 100): 15.43
  Eval Score: 18.9
  Epsilon: 0.367
  Memory size: 10,000
  Avg Loss: 0.0234
```

---

## 💡 Pro Tips

1. **Start small:** Use `qlearning_quick.yaml` for fast iteration
2. **Watch agents:** Visual feedback helps understand behavior
3. **Compare metrics:** Use `--compare` to see relative performance
4. **Adjust configs:** Modify YAML files for experiments
5. **Save outputs:** All results go to `reports/` and `checkpoints/`
6. **Use Makefile:** Shorter commands for common tasks
7. **Test first:** Run `test_all_functionality.py` after changes

---

## 🚀 Next Steps

After running these commands, you'll have:
- ✅ Verified functionality
- ✅ Baseline agent metrics
- ✅ Trained RL models
- ✅ Comparison visualizations
- ✅ Publication-ready plots

Ready to customize, experiment, and showcase your work!
