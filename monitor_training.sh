#!/bin/bash
# Monitor training progress

echo "=========================================="
echo "SNAKESENSE - Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
TRAINING_PIDS=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAINING_PIDS" ]; then
    echo "✓ Training complete (no train.py process running)"
    echo ""
else
    echo "🏃 Training in progress (PIDs: $TRAINING_PIDS)"
    echo ""
fi

# Check for checkpoint files
echo "Checkpoint Files:"
echo "----------------------------------------"
if [ -f "checkpoints/qlearning_metrics.json" ]; then
    echo "✓ qlearning_metrics.json ($(ls -lh checkpoints/qlearning_metrics.json | awk '{print $5}'))"

    # Show progress
    EPISODES=$(cat checkpoints/qlearning_metrics.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('episode_scores', [])))" 2>/dev/null)
    if [ ! -z "$EPISODES" ]; then
        echo "  Episodes completed: $EPISODES"

        # Show recent scores
        RECENT_SCORES=$(cat checkpoints/qlearning_metrics.json | python3 -c "import sys, json; data=json.load(sys.stdin); scores=data.get('episode_scores', []); print(f'{sum(scores[-10:])/len(scores[-10:]):.2f}' if len(scores)>=10 else 'N/A')" 2>/dev/null)
        if [ "$RECENT_SCORES" != "N/A" ]; then
            echo "  Avg score (last 10): $RECENT_SCORES"
        fi
    fi
else
    echo "⏳ No metrics file yet (training just started)"
fi

if [ -f "checkpoints/qlearning_final.pkl" ]; then
    echo "✓ qlearning_final.pkl ($(ls -lh checkpoints/qlearning_final.pkl | awk '{print $5}'))"
fi

if [ -f "checkpoints/dqn_metrics.json" ]; then
    echo "✓ dqn_metrics.json ($(ls -lh checkpoints/dqn_metrics.json | awk '{print $5}'))"
fi

if [ -f "checkpoints/dqn_final.pt" ]; then
    echo "✓ dqn_final.pt ($(ls -lh checkpoints/dqn_final.pt | awk '{print $5}'))"
fi

echo ""
echo "Other Output Files:"
echo "----------------------------------------"

if [ -f "reports/baseline_results.json" ]; then
    echo "✓ reports/baseline_results.json"
fi

if [ -f "reports/baseline_comparison.png" ]; then
    echo "✓ reports/baseline_comparison.png"
fi

echo ""
echo "=========================================="
echo "Commands:"
echo "----------------------------------------"
echo "Watch progress:   watch -n 5 ./monitor_training.sh"
echo "Check process:    ps aux | grep train.py"
echo "View metrics:     cat checkpoints/*_metrics.json | python3 -m json.tool"
echo "Kill training:    pkill -f train.py"
echo "=========================================="
