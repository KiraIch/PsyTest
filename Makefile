.PHONY: setup check generate batch test clean

setup:
	pip install -r requirements.txt

check:
	python -m src.backend.detector
	python scripts/validate_setup.py

generate:
	python scripts/generate_session.py --persona data/personas/mdd_severe.yaml --scenario data/scenarios/initial_session.yaml

batch:
	python scripts/batch_generate.py --config configs/default.yaml

test:
	pytest tests/ -v

clean:
	rm -rf data/output/sessions/* data/output/audio/* data/output/annotations/*
	rm -rf logs/*.log
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
