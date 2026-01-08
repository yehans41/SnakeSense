#!/bin/bash
# Quick demo script to showcase SnakeSense capabilities

echo "========================================="
echo "SNAKESENSE - Quick Demo"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating venv: source venv/bin/activate"
    echo ""
fi

# Test environment
echo "1. Testing environment..."
python3 test_environment.py || {
    echo "❌ Tests failed. Please install dependencies:"
    echo "   pip install -r requirements.txt"
    exit 1
}

echo ""
echo "2. Evaluating baseline agents (100 episodes each)..."
python3 experiments/evaluate.py --episodes 100 --output reports/baseline_results.json

echo ""
echo "3. Plotting baseline results..."
python3 metrics/plot_results.py --baseline reports/baseline_results.json --output-dir reports

echo ""
echo "========================================="
echo "✓ Demo Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - reports/baseline_results.json"
echo "  - reports/baseline_comparison.png"
echo ""
echo "Next steps:"
echo "  • Play manually: python3 play.py"
echo "  • Watch AI play: python3 play.py --mode agent --agent astar --games 3"
echo "  • Train Q-Learning: python3 experiments/train.py --config configs/qlearning.yaml"
echo "  • Train DQN: python3 experiments/train.py --config configs/dqn.yaml"
echo ""
