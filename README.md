# 🔍 Clearshield

A machine learning system for detecting fraudulent transactions using traditional ML algorithms and LSTM neural networks.

## 📋 Overview

This fraud detection system combines traditional machine learning with deep learning approaches to identify fraudulent transactions. The system features a hybrid architecture that leverages both statistical features and sequential patterns.

## 📁 Project Structure

```
Clearshield/
├── data/
│   ├── raw/                     # Original datasets (00)
│   ├── cleaned/                 # Cleaned datasets (01)
│   ├── processed/               # Processed datasets (02+03) (transaction files per user after matching fraud event, which should be used for model)
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
│   │   │   ├── 03a_transaction_type_clustering/ # Cluster types
│   │   │   ├── 03b_description_encoding/        # Process description
│   │   │   └── feature_pipeline.py
│   │   ├── 04_vulnerability_scanner/ # Security protection
│   │   └── pipeline.ipynb       # Main preprocessing pipeline (raw -> processed)
│   │
│   └── models/                  # Model training and evaluation
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
