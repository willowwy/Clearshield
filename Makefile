.PHONY: help setup install clean clean-train clean-pred clean-all \
        run run-train run-pred \
        test

# Default target
.DEFAULT_GOAL := help

help:
	@echo "======================================================================"
	@echo "ClearShield - Fraud Detection System"
	@echo "======================================================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - Initialize project (create dirs + install deps)"
	@echo "  make install        - Install Python dependencies"
	@echo ""
	@echo "Training Pipeline:"
	@echo "  make run-train      - Run training pipeline (Stage 1-5)"
	@echo "  make clean-train    - Clean training data + scan results"
	@echo ""
	@echo "Prediction Pipeline:"
	@echo "  make run-pred       - Run prediction pipeline (Stage 1-4)"
	@echo "  make clean-pred     - Clean prediction data"
	@echo ""
	@echo "General Commands:"
	@echo "  make run            - Alias for 'make run-train'"
	@echo "  make clean          - Clean all data directories"
	@echo "  make help           - Show this help message"
	@echo "======================================================================"

# ============================================================================
# Setup
# ============================================================================

setup:
	@echo "Setting up ClearShield project..."
	@python setup.py
	@echo "✓ Setup complete"

install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ============================================================================
# Pipeline Execution
# ============================================================================

run-train:
	@echo "Running TRAINING data pipeline..."
	@cd src/data_preprocess && python run_train_pipeline.py

run-pred:
	@echo "Running PREDICTION data pipeline..."
	@cd src/data_preprocess && python run_pred_pipeline.py

# Alias: run -> run-train (default)
run: run-train

# ============================================================================
# Cleaning
# ============================================================================

clean-train:
	@echo "Cleaning TRAINING data directories..."
	@rm -rf data/train/cleaned/*
	@rm -rf data/train/clustered_out/*
	@rm -rf data/train/by_member/*
	@rm -rf data/train/final/*
	@rm -f vulnerability_scan_results.json
	@echo "✓ Training data cleaned"

clean-pred:
	@echo "Cleaning PREDICTION data directories..."
	@rm -rf data/pred/cleaned/*
	@rm -rf data/pred/clustered_out/*
	@rm -rf data/pred/by_member/*
	@rm -rf data/pred/final/*
	@echo "✓ Prediction data cleaned"

clean-all: clean-train clean-pred
	@echo "✓ All data directories cleaned"

# Alias: clean -> clean-all (default)
clean: clean-all

# ============================================================================
# Testing (placeholder for future tests)
# ============================================================================

test:
	@echo "Running tests..."
	@echo "TODO: Implement test suite"
