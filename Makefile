.PHONY: setup install clean run help

help:
	@echo "ClearShield - Fraud Detection System"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup    - Initialize project (create dirs + install deps)"
	@echo "  make install  - Install dependencies only"
	@echo "  make run      - Run complete data processing pipeline"
	@echo "  make clean    - Clean generated data directories"
	@echo "  make help     - Show this help message"

setup:
	@python setup.py

install:
	@pip install -r requirements.txt

run:
	@cd src/data_preprocess && python run_pipeline.py

clean:
	@echo "Cleaning data directories..."
	@rm -rf data/cleaned/* data/clustered_out/* data/by_member/* data/processed/* data/final/*
	@echo "✓ Data directories cleaned"
