# 🔍 Clearshield

A machine learning system for detecting fraudulent transactions using traditional ML algorithms and LSTM neural networks.

## 📋 Overview

This fraud detection system combines traditional machine learning with deep learning approaches to identify fraudulent transactions. The system features a hybrid architecture that leverages both statistical features and sequential patterns.

## 📁 Project Structure

```
ClearShield/
├── data/
│   ├── raw/                     # Original datasets (00)
│   ├── cleaned/                 # Cleaned datasets (01)
│   ├── processed/               # Processed datasets (02+03)
│   │                            # Transaction files per user after matching fraud events
│   │                            # Ready for model training
│   └── external/                # External data sources (optional)
│
├── docs/                        # Documentation files
│
├── notebooks/                   # Jupyter notebooks for analysis
│
├── src/
│   ├── data_preprocess/
│   │   ├── 01_data_cleaning/       # Data cleaning scripts
│   │   ├── 02_fraud_relabeling/    # Fraud label adjustment
│   │   ├── 03_feature_engineering/ # Feature refinement
│   │   │   ├── 03a_transaction_type_clustering/ # Cluster transaction types
│   │   │   ├── 03b_description_encoding/        # Process descriptions
│   │   │   └── feature_pipeline.py              # Feature engineering pipeline
│   │   ├── 04_encoding/            # Data encoding
│   │   ├── 05_vulnerability_scanner/ # Security protection
│   │   └── pipeline.ipynb          # Main preprocessing pipeline (raw → processed)
│   │
│   └── models/
│       ├── preprocessing/          # Model-specific preprocessing
│       └── __init__.py
│
├── config/                      # Configuration files
│
├── .venv/                       # Virtual environment
├── venv/                        # Alternative virtual environment
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```



## 🚀 Quick Start

1. **Clone the repository**

```bash
   git clone <repository-url>
   cd ClearShield
```

2. **Set up virtual environment**

```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
```

3. **Run the preprocessing pipeline**

- Open `src/data_preprocess/pipeline.ipynb`
- Execute cells sequentially to transform raw data into processed datasets

4. **Train models**

- Navigate to `src/models/`
- Follow model-specific training instructions
