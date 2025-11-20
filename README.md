# 🔍 Clearshield

A machine learning system for detecting fraudulent transactions using traditional ML algorithms and LSTM neural networks.

## 📋 Overview

This fraud detection system combines traditional machine learning with deep learning approaches to identify fraudulent transactions. The system features a hybrid architecture that leverages both statistical features and sequential patterns.

## 📁 Project Structure

```
ClearShield/
├── data/
│   ├── train/                   # Training data pipeline
│   │   ├── raw/                 # Original datasets
│   │   ├── cleaned/             # Cleaned datasets (Stage 1)
│   │   ├── clustered_out/       # Clustered datasets (Stage 2)
│   │   ├── by_member/           # Fraud-matched datasets (Stage 3)
│   │   │   ├── temp/            # Temporary (auto-deleted)
│   │   │   ├── matched/         # Members with matched fraud
│   │   │   ├── unmatched/       # Members with unmatched fraud
│   │   │   ├── no_fraud/        # Members without fraud
│   │   │   └── member_summary.csv
│   │   └── final/               # Final encoded datasets (Stage 4)
│   │       ├── matched/         # Ready for model training
│   │       ├── unmatched/
│   │       └── no_fraud/
│   │
│   ├── pred/                    # Prediction data pipeline
│   │   ├── raw/                 # New transaction data
│   │   ├── cleaned/             # Cleaned (Stage 1)
│   │   ├── clustered_out/       # Clustered (Stage 2 - inference)
│   │   ├── by_member/           # Reorganized by member (Stage 3)
│   │   └── final/               # Encoded for prediction (Stage 4)
│   │
│   └── external/                # External data sources (optional)
│
├── docs/                        # Documentation files
│
├── notebooks/                   # Jupyter notebooks for analysis
│
├── src/
│   ├── data_preprocess/
│   │   ├── 01_data_cleaning/        # Step 1: Data cleaning scripts
│   │   ├── 02_feature_engineering/  # Step 2: Feature engineering
│   │   │   ├── 02a_transaction_type_clustering/ # Cluster transaction types
│   │   │   ├── 02b_description_encoding/        # BERT encoding + clustering
│   │   │   └── 02_feature_engineering.py        # Main pipeline
│   │   ├── 03_fraud_relabeling/     # Step 3: Fraud matching and re-labeling
│   │   ├── 04_encoding/             # Step 4: Feature encoding
│   │   ├── 05_vulnerability_scanner/ # Security protection
│   │   └── pipeline.ipynb           # Complete preprocessing pipeline
│   │
│   └── models/
│       ├── preprocessing/           # Model-specific preprocessing
│       └── __init__.py
│
├── config/                      # Configuration files
│   └── tokenize_dict.json       # Categorical encoding dictionary
│
├── .venv/                       # Virtual environment
├── venv/                        # Alternative virtual environment
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## 🔄 Data Processing Pipeline

The preprocessing pipeline consists of 4 sequential stages:

### Step 1: Data Cleaning (`01_data_cleaning`)
- Standardize CSV headers
- Fix comma and formatting issues
- Clean Amount field (remove $, convert to numeric)
- Fill missing values
- Rename files based on date range

**Data Flow**: `train/raw/` → `train/cleaned/`

### Step 2: Feature Engineering (`02_feature_engineering`)
- Encode transaction descriptions using BERT-tiny (`prajjwal1/bert-tiny`)
- Apply PCA dimensionality reduction (default: 20 dimensions)
- Perform automatic clustering (MiniBatchKMeans, k=60)
- Add cluster_id column
- **Configurable paths**: Supports custom input/output directories via parameters

**Data Flow**: `train/cleaned/` → `train/clustered_out/`

### Step 3: Fraud Matching (`03_fraud_relabeling`)
- **Stage 1**: Reorganize transactions by Member ID
- **Stage 2**: Match fraud adjustments to original transactions
  - Extract dates from fraud descriptions
  - Match by amount and time window (30 days)
  - Prioritize "Mobile Deposit" transactions
- Filter members with ≥10 transactions (configurable)
- Categorize into matched/unmatched/no_fraud

**Data Flow**: `train/clustered_out/` → `train/by_member/` → `train/processed/[matched|unmatched|no_fraud]/`

### Step 4: Feature Encoding (`04_encoding`)
- Remove ID columns (Account ID, Member ID)
- Encode categorical features (Account Type, Action Type, Source Type, Product ID)
- Parse time features to `time` objects (HH:MM:SS format)
- Convert date features to datetime
- Remove text columns (Transaction Description, Fraud Adjustment Indicator)

**Data Flow**: `train/processed/` → `train/final/[matched|unmatched|no_fraud]/`

**Final Output**: `data/train/final/` contains model-ready datasets

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone <repository-url>
cd ClearShield
```

2. **Set up environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python setup.py            # Creates directories and installs dependencies
```

Or use Makefile:
```bash
make setup
```

3. **Run the preprocessing pipeline**

**Option A: Automated Script (Recommended)**
```bash
cd src/data_preprocess
python run_pipeline.py
```

Or using Makefile:
```bash
make run
```

Advanced usage:
```bash
python run_pipeline.py --help                    # Show all options
python run_pipeline.py --skip-cleaning           # Skip data cleaning
python run_pipeline.py --min-history 15          # Set minimum history to 15
```

**Option B: Jupyter Notebook (For exploration)**
- Open `src/data_preprocess/pipeline.ipynb`
- Execute cells sequentially to run the complete 4-stage pipeline

**Final datasets** will be in `data/train/final/[matched|unmatched|no_fraud]/`

4. **Train models**

- Navigate to `src/models/`
- Use datasets from `data/train/final/[matched|unmatched|no_fraud]/`
- Follow model-specific training instructions

## 🔧 Advanced Configuration

### Custom Data Paths
You can override default paths programmatically:

```python
from src.data_preprocess.feature_engineering import run_stage2

run_stage2(
    processed_dir='/custom/path/to/cleaned',
    output_dir='/custom/path/to/output',
    model_name='prajjwal1/bert-tiny',
    pca_dim=20,
    max_k=60,
    verbose=True
)
```

### Pipeline Parameters
- `--min-history N`: Minimum transaction count per member (default: 10)
- `--skip-cleaning`: Skip data cleaning stage
- `--skip-feature-engineering`: Skip feature engineering stage
- `--skip-fraud-matching`: Skip fraud matching stage
- `--skip-encoding`: Skip feature encoding stage
- `--quiet`: Suppress verbose output

## 🔮 Prediction Pipeline (Inference Mode)

For processing new data without retraining the clustering model:

### 1. Train Mode (One-time Setup)
First, run the training pipeline to create and save the clustering model:

```bash
cd src/data_preprocess
python run_pipeline.py  # Saves cluster_model.pkl
```

This will save `cluster_model.pkl` containing:
- Pre-trained PCA transformer
- Fitted KMeans clustering model
- BERT model configuration

### 2. Inference Mode (For New Data)
Apply the saved model to new data without retraining:

**Single CSV file:**
```bash
cd src/data_preprocess/02_feature_engineering
python inference_stage2.py \
  --input /path/to/new/data.csv \
  --output /path/to/output/data.csv \
  --model cluster_model.pkl
```

**Entire directory:**
```bash
python inference_stage2.py \
  --input /path/to/new/cleaned/ \
  --output /path/to/new/clustered_out/ \
  --model cluster_model.pkl
```

**Parameters:**
- `--input`: Input CSV file or directory
- `--output`: Output CSV file or directory
- `--model`: Path to saved model (default: `cluster_model.pkl`)
- `--text-column`: Column name for transaction descriptions (default: `Transaction Description`)
- `--batch-size`: Encoding batch size (default: 512)
- `--quiet`: Suppress progress output

### 3. Complete Prediction Pipeline
For end-to-end processing of new data:

```bash
# Step 1: Clean new data
python run_pipeline.py --skip-feature-engineering --skip-fraud-matching --skip-encoding

# Step 2: Apply clustering (inference mode)
python 02_feature_engineering/inference_stage2.py \
  --input ../../data/predict/cleaned \
  --output ../../data/predict/clustered_out

# Step 3: Continue with fraud matching and encoding
python run_pipeline.py --skip-cleaning --skip-feature-engineering
```
