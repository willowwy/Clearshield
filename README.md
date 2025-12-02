# 🔍 Clearshield

A machine learning system for detecting fraudulent transactions using traditional ML algorithms and LSTM neural networks.

## 📋 Overview

This fraud detection system combines traditional machine learning with deep learning approaches to identify fraudulent transactions. The system features a hybrid architecture that leverages both statistical features and sequential patterns.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd ClearShield

# Setup (create directories + install dependencies)
make setup
```

Or manually:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python setup.py
```

### 2. Data Preprocessing

Place your raw data in `data/train/raw/` or `data/pred/raw/`, then run:

```bash
# Training data pipeline (5 stages)
make run-train

# Prediction data pipeline (4 stages)
make run-pred
```

Or use Python directly:
```bash
cd src/data_preprocess
python run_train_pipeline.py          # Training mode
python run_pred_pipeline.py           # Prediction mode
```

### 3. Model Training

```bash
# Train sequence model (Step 1)
make train-seq

# Train judge model (Step 2)
make train-judge

# Or train both models sequentially
make train-all
```

Advanced options:
```bash
# Custom training parameters
make train-seq EPOCHS=200 MAX_LEN=100 SAVE_DIR=my_checkpoints
```

### 4. Model Inference

```bash
# Run inference on a single member
make infer MEMBER_ID=12345

# Use custom data folder
make infer MEMBER_ID=12345 INFER_FOLDER=data/pred/final
```

Or use Python directly:
```bash
cd src/models
python inference.py \
  --folder ../../data/train/final/matched \
  --member_id 12345 \
  --sequence_model_path ../../checkpoints/best_model_enc.pth \
  --judge_model_path ../../checkpoints/best_judge_model.pth \
  --max_len 50
```

### 5. View All Commands

```bash
make help
```

## 🔄 Data Processing Pipeline

The system processes data through 5 sequential stages:

### Training Pipeline (5 Stages)
1. **Data Cleaning** - Standardize headers, fix formatting, clean Amount field
2. **Feature Engineering** - BERT encoding, PCA reduction, clustering (k=60)
3. **Fraud Matching** - Reorganize by member, match fraud adjustments, categorize
4. **Feature Encoding** - Encode categorical features, parse time/date fields
5. **Vulnerability Scanning** (Optional) - Security testing, adversarial attack detection

**Data Flow**: `raw/` → `cleaned/` → `clustered_out/` → `by_member/` → `final/`

### Prediction Pipeline (4 Stages)
Same as training pipeline but uses pre-trained clustering models (Stages 1-4 only).

**Output**: Model-ready datasets in `data/train/final/` or `data/pred/final/`

## 🔧 Configuration

### Pipeline Parameters

**Training Pipeline:**
```bash
python run_train_pipeline.py --help                    # Show all options
python run_train_pipeline.py --min-history 15          # Minimum 15 transactions per member
python run_train_pipeline.py --skip-vuln-scan          # Skip security scanning
python run_train_pipeline.py --vuln-sample-size 2000   # Use 2000 samples for scanning
```

**Model Training:**
```bash
# Configurable via Makefile variables
EPOCHS=110              # Training epochs (default: 110)
MAX_LEN=50              # Maximum sequence length (default: 50)
SAVE_DIR=checkpoints    # Model save directory (default: checkpoints)
```

### Custom Data Paths

Override paths programmatically:
```python
from src.data_preprocess.feature_engineering import run_stage2

run_stage2(
    processed_dir='/custom/path/to/cleaned',
    output_dir='/custom/path/to/output',
    model_name='prajjwal1/bert-tiny',
    pca_dim=20,
    max_k=60
)
```

## 🔮 Inference Mode

For processing new data using pre-trained models:

### Using Prediction Pipeline
```bash
# Place new data in data/pred/raw/
make run-pred

# Run inference
make infer MEMBER_ID=12345 INFER_FOLDER=data/pred/final
```

### Manual Stage 2 Inference
```bash
cd src/data_preprocess/02_feature_engineering
python inference_stage2.py \
  --input ../../data/pred/cleaned \
  --output ../../data/pred/clustered_out \
  --model cluster_model.pkl
```

## 🧹 Cleaning

```bash
make clean-train      # Clean training data
make clean-pred       # Clean prediction data
make clean-models     # Remove model checkpoints
make clean            # Clean all data directories
```

## 📁 Project Structure

```
ClearShield/
├── data/
│   ├── train/                   # Training data pipeline
│   │   ├── raw/                 # Original datasets
│   │   ├── cleaned/             # Cleaned datasets (Stage 1)
│   │   ├── clustered_out/       # Clustered datasets (Stage 2)
│   │   ├── by_member/           # Fraud-matched datasets (Stage 3)
│   │   │   ├── temp/            # Temporary files
│   │   │   ├── matched/         # Fraud matched members
│   │   │   ├── unmatched/       # Fraud unmatched members
│   │   │   └── no_fraud/        # No fraud members
│   │   └── final/               # Final encoded datasets (Stage 4)
│   │       ├── matched/         # Model-ready
│   │       ├── unmatched/
│   │       └── no_fraud/
│   │
│   └── pred/                    # Prediction data pipeline
│       ├── raw/                 # New transaction data
│       ├── cleaned/
│       ├── clustered_out/
│       ├── by_member/
│       └── final/
│
├── src/
│   ├── data_preprocess/
│   │   ├── 01_data_cleaning/
│   │   ├── 02_feature_engineering/
│   │   │   ├── 02a_transaction_type_clustering/
│   │   │   └── 02b_description_encoding/
│   │   ├── 03_fraud_relabeling/
│   │   ├── 04_encoding/
│   │   ├── 05_security/
│   │   ├── train_pipeline.ipynb
│   │   ├── pred_pipeline.ipynb
│   │   ├── run_train_pipeline.py
│   │   └── run_pred_pipeline.py
│   │
│   └── models/
│       ├── backbone_model.py    # Sequence model
│       ├── judge.py             # Fraud judge model
│       ├── datasets.py          # Data loaders
│       ├── inference.py         # Inference script
│       ├── load_model.py        # Model loading utilities
│       └── loss.py              # Loss functions
│
├── config/
│   ├── pipeline_config.py       # Centralized path configuration
│   └── tokenize_dict.json       # Categorical encoding dictionary
│
├── checkpoints/                 # Trained model checkpoints
├── notebooks/                   # Jupyter notebooks for analysis
├── docs/                        # Documentation
│
├── train.py                     # Sequence model training
├── train_judge.py               # Judge model training
├── Makefile                     # Automation commands
├── setup.py                     # Project setup script
├── requirements.txt             # Python dependencies
└── README.md
```

## 📦 Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch ≥2.2.0
- Transformers ≥4.38.0
- scikit-learn ≥1.3.0
- pandas ≥2.0.0
- numpy ≥1.24.0
- joblib ≥1.3.0
- cryptography ≥41.0.0

## 🔐 Security

Stage 5 vulnerability scanning includes:
- Data poisoning detection
- Adversarial attack testing (FGSM, PGD)
- Privacy attack simulation
- Automated security reporting

Output: `vulnerability_scan_results.json`
