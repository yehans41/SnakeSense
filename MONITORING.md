# Training Monitoring Guide

How to monitor training progress and know when it's complete.

## 🎯 Quick Answer

Training is complete when:
1. ✅ The `train.py` process is no longer running
2. ✅ You see "Training complete!" in the output
3. ✅ Final checkpoint files exist in `checkpoints/`
4. ✅ Metrics JSON file shows all episodes completed

---

## 📊 Method 1: Use the Training Monitor (EASIEST)

### Run Once
```bash
./monitor_training.sh
```

**Output shows:**
- ✅ Training status (running or complete)
- ✅ Process IDs if running
- ✅ Checkpoint files created
- ✅ Episodes completed
- ✅ Recent average scores

### Watch Continuously (Auto-refresh every 5 seconds)
```bash
watch -n 5 ./monitor_training.sh
```
Press `Ctrl+C` to stop watching.

---

## 🔍 Method 2: Check Process Status

### Check if Training is Running
```bash
ps aux | grep "train.py" | grep -v grep
```

**If training is running:**
```
yehan  12938  96.9  0.2  ....  Python experiments/train.py ...
```

**If training is complete:**
```
(no output)
```

### Get Process ID
```bash
pgrep -f "train.py"
```

### Kill Training (if needed)
```bash
pkill -f "train.py"
```

---

## 📁 Method 3: Check Output Files

### List Checkpoint Files
```bash
ls -lh checkpoints/
```

**During training:**
- May be empty initially
- `*_ep250.pkl` or `*_ep250.pt` (checkpoint every 250 episodes)

**After training:**
- `qlearning_final.pkl` (Q-Learning)
- `qlearning_metrics.json` (training history)
- `dqn_final.pt` (DQN)
- `dqn_metrics.json` (training history)

### Check Metrics File
```bash
# See if metrics file exists
ls -lh checkpoints/qlearning_metrics.json

# View formatted metrics
cat checkpoints/qlearning_metrics.json | python3 -m json.tool | less

# Count episodes completed
cat checkpoints/qlearning_metrics.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'Episodes: {len(data[\"episode_scores\"])}')"

# Show recent progress
cat checkpoints/qlearning_metrics.json | python3 -c "import sys, json; data=json.load(sys.stdin); scores=data['episode_scores'][-10:]; print(f'Last 10 episodes avg score: {sum(scores)/len(scores):.2f}')"
```

---

## ⏱️ Method 4: Estimate Time Remaining

### Expected Training Times

| Config | Episodes | Board Size | Est. Time |
|--------|----------|------------|-----------|
| `qlearning_quick.yaml` | 500 | 8×8 | 2-3 min |
| `qlearning.yaml` | 5,000 | 8×8 | 10-15 min |
| `dqn.yaml` | 10,000 | 10×10 | 30-60 min |

**Factors affecting speed:**
- Board size (larger = slower)
- Episode length (longer = slower)
- CPU speed
- Whether GPU is used (DQN only)

### Calculate Progress
```bash
# If training 500 episodes:
# Check current episode count, divide by 500

python3 -c "
import json
with open('checkpoints/qlearning_metrics.json') as f:
    data = json.load(f)
episodes = len(data['episode_scores'])
total = 500  # or whatever your config specifies
percent = (episodes / total) * 100
print(f'Progress: {episodes}/{total} episodes ({percent:.1f}%)')
"
```

---

## 📝 Method 5: Read Training Output

### Find Output (if running in background)
The training output is captured in a temp file. To find it:

```bash
# Find the most recent training output
ls -lt /tmp/claude/-Users-yehan-Desktop-SnakeSense/tasks/*.output 2>/dev/null | head -1
```

### View Full Output
```bash
# Replace with actual path
cat /path/to/output/file
```

### Watch Output in Real-Time
```bash
tail -f /path/to/output/file
```

---

## ✅ Signs Training is Complete

### 1. Terminal Output
Look for this message:
```
Training complete!
Total time: 123.4s
Final model saved to: checkpoints/qlearning_final.pkl
Metrics saved to: checkpoints/qlearning_metrics.json
```

### 2. Final Files Exist
```bash
ls checkpoints/
```
Should show:
- `qlearning_final.pkl` or `dqn_final.pt`
- `qlearning_metrics.json` or `dqn_metrics.json`

### 3. Process Not Running
```bash
ps aux | grep train.py | grep -v grep
```
Should return nothing.

### 4. Metrics Show All Episodes
```bash
cat checkpoints/qlearning_metrics.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
episodes = len(data['episode_scores'])
print(f'Episodes completed: {episodes}')
"
```

Compare to config file:
```bash
grep "episodes:" configs/qlearning_quick.yaml
# Should show: episodes: 500
```

---

## 🔔 Method 6: Get Notification When Done

### Simple Shell Script
```bash
# Wait for training to finish, then notify
while pgrep -f "train.py" > /dev/null; do
    sleep 10
done
echo "Training complete!"
say "Training complete" 2>/dev/null  # macOS text-to-speech
```

### Run in Background
```bash
(while pgrep -f "train.py" > /dev/null; do sleep 10; done && echo "✓ Training done!" && say "Training complete") &
```

---

## 🎬 After Training Completes

### 1. Verify Completion
```bash
./monitor_training.sh
```

### 2. Check Results
```bash
# View training metrics
cat checkpoints/qlearning_metrics.json | python3 -m json.tool | less

# Get summary stats
python3 -c "
import json
with open('checkpoints/qlearning_metrics.json') as f:
    data = json.load(f)
scores = data['episode_scores']
print(f'Total episodes: {len(scores)}')
print(f'Final avg score (last 100): {sum(scores[-100:])/100:.2f}')
print(f'Max score: {max(scores)}')
print(f'Training time: {data[\"training_time\"]:.1f}s')
"
```

### 3. Plot Results
```bash
python3 metrics/plot_results.py \
  --metrics checkpoints/qlearning_metrics.json \
  --output-dir reports
```

### 4. Compare with Baselines
```bash
python3 metrics/plot_results.py \
  --baseline reports/baseline_results.json \
  --metrics checkpoints/qlearning_metrics.json \
  --output-dir reports
```

---

## 🐛 Troubleshooting

### Training Seems Stuck
```bash
# Check if process is actually running
ps aux | grep train.py

# Check CPU usage (should be high)
top -pid $(pgrep -f train.py)

# Force stop if needed
pkill -9 -f train.py
```

### No Output Files
- Training may not have reached first checkpoint (default every 250 episodes)
- Check if process crashed: `echo $?` (0 = success, non-zero = error)
- Look for error messages in output

### Training Too Slow
- Reduce episodes in config file
- Use smaller board size
- Use `qlearning_quick.yaml` instead

### Want to Resume Training
Q-Learning and DQN agents support save/load. Modify training script to:
```python
# Load previous checkpoint
if os.path.exists('checkpoints/qlearning_final.pkl'):
    agent.load('checkpoints/qlearning_final.pkl')
```

---

## 📊 Current Training Status

Run this command to get current status:
```bash
./monitor_training.sh
```

Or add it to your shell:
```bash
alias train-status='cd ~/Desktop/SnakeSense && ./monitor_training.sh'
```

Then just type: `train-status`

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Check status | `./monitor_training.sh` |
| Watch continuously | `watch -n 5 ./monitor_training.sh` |
| Check if running | `pgrep -f train.py` |
| Stop training | `pkill -f train.py` |
| View metrics | `cat checkpoints/*_metrics.json \| python3 -m json.tool` |
| List outputs | `ls -lh checkpoints/ reports/` |
| Plot results | `python3 metrics/plot_results.py --metrics checkpoints/qlearning_metrics.json` |

---

## 💡 Pro Tips

1. **Use the monitor script** - It's the easiest way: `./monitor_training.sh`

2. **Watch in another terminal** - While training runs:
   ```bash
   watch -n 5 ./monitor_training.sh
   ```

3. **Check before leaving** - Make sure training started:
   ```bash
   sleep 30 && ./monitor_training.sh
   ```

4. **Background training** - Free up your terminal:
   ```bash
   nohup python3 experiments/train.py --config configs/qlearning.yaml > training.log 2>&1 &
   ```

5. **Email notification** (if you have mail configured):
   ```bash
   python3 experiments/train.py --config configs/dqn.yaml && echo "Done!" | mail -s "Training Complete" your@email.com
   ```

---

Training monitoring made easy! 🎉
