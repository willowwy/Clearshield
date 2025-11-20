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
	@echo "Pipeline Commands:"
	@echo "  make run-train      - Run training data pipeline (Stage 1-4)"
	@echo "  make run-pred       - Run prediction data pipeline (Stage 1-4, inference)"
	@echo "  make run            - Alias for 'make run-train'"
	@echo ""
	@echo "Cleaning Commands:"
	@echo "  make clean-train    - Clean training data directories"
	@echo "  make clean-pred     - Clean prediction data directories"
	@echo "  make clean-all      - Clean both training and prediction directories"
	@echo "  make clean          - Alias for 'make clean-all'"
	@echo ""
	@echo "Other Commands:"
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
