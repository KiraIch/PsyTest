.PHONY: setup check generate batch report test clean

PERSONA ?= data/personas/mdd_severe.yaml
SCENARIO ?= data/scenarios/initial_session.yaml
CONFIG ?= configs/default.yaml

setup:
	pip install -r requirements.txt

check:
	python -m src.backend.detector
	python scripts/validate_setup.py

generate:
	python scripts/generate_session.py --persona $(PERSONA) --scenario $(SCENARIO)

batch:
	python scripts/batch_generate.py --config $(CONFIG)

report:
	python scripts/generate_report.py --input $(INPUT) --output $(OUTPUT)

test:
	pytest tests/ -v

clean:
	rm -rf data/output/sessions/* data/output/audio/* data/output/annotations/*
	rm -rf logs/*.log
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
