# 🔍 Clearshield

A machine learning system for detecting fraudulent transactions using traditional ML algorithms and LSTM neural networks.

## 📋 Overview

This fraud detection system combines traditional machine learning with deep learning approaches to identify fraudulent transactions. The system features a hybrid architecture that leverages both statistical features and sequential patterns.

## 📁 Project Structure

```
Clearshield/
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/               # Processed datasets
│   │   ├── features/           # Feature files
│   │   └── models/             # Model outputs
│   └── external/               # External data sources
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading utilities
│   │   ├── data_cleaner.py     # Data cleaning functions
│   │   └── data_validator.py   # Data validation logic
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base_features.py    # Basic feature extraction
│   │   ├── time_features.py    # Temporal features
│   │   ├── amount_features.py  # Transaction amount features
│   │   ├── merchant_features.py # Merchant-based features
│   │   ├── user_features.py    # User behavioral features
│   │   ├── sequence_features.py # LSTM sequence features
│   │   └── feature_pipeline.py # Feature processing pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # Model base class
│   │   ├── traditional_ml.py   # Traditional ML models
│   │   ├── lstm_model.py       # LSTM neural network
│   │   └── hybrid_model.py     # Hybrid ensemble model
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── helpers.py          # Helper functions
│   │   └── metrics.py          # Evaluation metrics
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # Visualization utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_evaluation.ipynb
│
├── config/
│   ├── feature_config.yaml     # Feature engineering config
│   ├── model_config.yaml       # Model parameters
│   └── data_config.yaml        # Data processing config
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_data.py
│
├── requirements.txt
├── setup.py
└── README.md
```
