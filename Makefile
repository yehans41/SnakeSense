.PHONY: install train eval plot clean test play

install:
	pip install -r requirements.txt

play:
	python play.py

train-random:
	python experiments/train.py --config configs/random.yaml

train-qlearning:
	python experiments/train.py --config configs/qlearning.yaml

train-dqn:
	python experiments/train.py --config configs/dqn.yaml

eval:
	python experiments/evaluate.py

plot:
	python metrics/plot_results.py

test:
	python -m pytest tests/

clean:
	rm -rf checkpoints/*.pt
	rm -rf reports/*.png
	rm -rf reports/*.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

baseline:
	python experiments/evaluate.py --agents random greedy astar

all-train:
	make train-qlearning
	make train-dqn
