.PHONY: help setup install clean clean-train clean-pred clean-all \
        run run-train run-pred \
        train-seq train-judge train-all infer \
        clean-models test

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Configuration Variables
# ============================================================================

# Model training parameters
EPOCHS ?= 110
MAX_LEN ?= 50
SAVE_DIR ?= checkpoints
SEQ_MODEL_PATH ?= $(SAVE_DIR)/best_model_enc.pth
JUDGE_MODEL_PATH ?= $(SAVE_DIR)/best_judge_model.pth

# Inference parameters
INFER_FOLDER ?= data/train/final/matched
MEMBER_ID ?=

help:
	@echo "======================================================================"
	@echo "ClearShield - Fraud Detection System"
	@echo "======================================================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - Initialize project (create dirs + install deps)"
	@echo "  make install        - Install Python dependencies"
	@echo ""
	@echo "Data Preprocessing Pipeline:"
	@echo "  make run-train      - Run training pipeline (Stage 1-5)"
	@echo "  make run-pred       - Run prediction pipeline (Stage 1-4)"
	@echo "  make run            - Alias for 'make run-train'"
	@echo ""
	@echo "Model Training:"
	@echo "  make train-seq      - Train sequence model (Step 1)"
	@echo "  make train-judge    - Train judge model (Step 2)"
	@echo "  make train-all      - Train both models sequentially"
	@echo ""
	@echo "Model Inference:"
	@echo "  make infer          - Run inference on a single member"
	@echo "                        Usage: make infer MEMBER_ID=12345"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean-train    - Clean training data + scan results"
	@echo "  make clean-pred     - Clean prediction data"
	@echo "  make clean-models   - Remove trained model checkpoints"
	@echo "  make clean          - Clean all data directories"
	@echo ""
	@echo "General:"
	@echo "  make test           - Run test suite"
	@echo "  make help           - Show this help message"
	@echo "======================================================================"
	@echo ""
	@echo "Configuration:"
	@echo "  EPOCHS=$(EPOCHS)        - Training epochs"
	@echo "  MAX_LEN=$(MAX_LEN)      - Maximum sequence length"
	@echo "  SAVE_DIR=$(SAVE_DIR)    - Model checkpoint directory"
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
# Data Preprocessing Pipeline
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
# Model Training
# ============================================================================

train-seq:
	@echo "Training sequence model..."
	@echo "  Epochs: $(EPOCHS)"
	@echo "  Max Length: $(MAX_LEN)"
	@echo "  Save Directory: $(SAVE_DIR)"
	@mkdir -p $(SAVE_DIR)
	@python train.py --save_model --epochs $(EPOCHS) --max_len $(MAX_LEN) --save_dir $(SAVE_DIR)
	@echo "✓ Sequence model training complete: $(SEQ_MODEL_PATH)"

train-judge:
	@echo "Training judge model..."
	@if [ ! -f "$(SEQ_MODEL_PATH)" ]; then \
		echo "Error: Sequence model not found at $(SEQ_MODEL_PATH)"; \
		echo "Please run 'make train-seq' first"; \
		exit 1; \
	fi
	@python train_judge.py --sequence_model_path $(SEQ_MODEL_PATH)
	@echo "✓ Judge model training complete: $(JUDGE_MODEL_PATH)"

train-all: train-seq train-judge
	@echo "======================================================================"
	@echo "✓ All models trained successfully!"
	@echo "  - Sequence model: $(SEQ_MODEL_PATH)"
	@echo "  - Judge model: $(JUDGE_MODEL_PATH)"
	@echo "======================================================================"

# ============================================================================
# Model Inference
# ============================================================================

infer:
	@if [ -z "$(MEMBER_ID)" ]; then \
		echo "Error: MEMBER_ID is required"; \
		echo "Usage: make infer MEMBER_ID=12345"; \
		echo "Optional: make infer MEMBER_ID=12345 INFER_FOLDER=data/pred/final"; \
		exit 1; \
	fi
	@if [ ! -f "$(SEQ_MODEL_PATH)" ]; then \
		echo "Error: Sequence model not found at $(SEQ_MODEL_PATH)"; \
		echo "Please run 'make train-seq' first"; \
		exit 1; \
	fi
	@if [ ! -f "$(JUDGE_MODEL_PATH)" ]; then \
		echo "Error: Judge model not found at $(JUDGE_MODEL_PATH)"; \
		echo "Please run 'make train-judge' first"; \
		exit 1; \
	fi
	@echo "Running inference for member $(MEMBER_ID)..."
	@cd src/models && python inference.py \
		--folder ../../$(INFER_FOLDER) \
		--member_id $(MEMBER_ID) \
		--sequence_model_path ../../$(SEQ_MODEL_PATH) \
		--judge_model_path ../../$(JUDGE_MODEL_PATH) \
		--max_len $(MAX_LEN)

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

clean-models:
	@echo "Cleaning model checkpoints..."
	@rm -rf $(SAVE_DIR)/*.pth
	@rm -rf $(SAVE_DIR)/*.json
	@echo "✓ Model checkpoints cleaned"

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
