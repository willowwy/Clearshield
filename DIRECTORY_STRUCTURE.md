# ClearShield - Directory Structure

## Overview

This document describes the complete directory structure for the ClearShield fraud detection system, including both training and prediction pipelines.

---

## 📁 Complete Directory Tree

```
Clearshield/
├── config/
│   └── tokenize_dict.json              # Feature encoding configuration
│
├── data/
│   ├── train/                          # Training Data Pipeline
│   │   ├── raw/                        # Stage 0: Raw input files
│   │   │   └── *.csv                   # Original transaction CSVs
│   │   │
│   │   ├── cleaned/                    # Stage 1: Data Cleaning
│   │   │   └── MM-DD-YYYY_to_MM-DD-YYYY.csv
│   │   │
│   │   ├── clustered_out/              # Stage 2: Feature Engineering
│   │   │   └── MM-DD-YYYY_to_MM-DD-YYYY.csv (with cluster_id)
│   │   │
│   │   ├── by_member/                  # Stage 3: Fraud Matching & Categorization
│   │   │   ├── temp/                   # Temporary: reorganized by member
│   │   │   │   └── member_*.csv        # (Auto-created, auto-deleted)
│   │   │   │
│   │   │   ├── matched/                # Fraud matching succeeded
│   │   │   │   └── member_*.csv
│   │   │   │
│   │   │   ├── unmatched/              # Fraud matching partially failed
│   │   │   │   └── member_*.csv
│   │   │   │
│   │   │   ├── no_fraud/               # No fraud adjustments
│   │   │   │   └── member_*.csv
│   │   │   │
│   │   │   └── member_summary.csv      # Statistics summary
│   │   │
│   │   └── final/                      # Stage 4: Feature Encoding
│   │       ├── matched/
│   │       │   └── member_*.csv
│   │       ├── unmatched/
│   │       │   └── member_*.csv
│   │       └── no_fraud/
│   │           └── member_*.csv
│   │
│   └── pred/                           # Prediction Data Pipeline
│       ├── raw/                        # Stage 0: Raw input files
│       │   └── *.csv                   # New transaction CSVs
│       │
│       ├── cleaned/                    # Stage 1: Data Cleaning
│       │   └── MM-DD-YYYY_to_MM-DD-YYYY.csv
│       │
│       ├── clustered_out/              # Stage 2: Clustering (Inference)
│       │   └── MM-DD-YYYY_to_MM-DD-YYYY.csv (with cluster_id)
│       │
│       ├── by_member/                  # Stage 3: Reorganize by Member
│       │   └── member_*.csv (with Fraud=0)
│       │
│       └── final/                      # Stage 4: Feature Encoding
│           └── member_*.csv
│
├── src/
│   └── data_preprocess/
│       ├── 01_data_cleaning/
│       │   └── 01_data_cleaning.py
│       │
│       ├── 02_feature_engineering/
│       │   ├── 02_feature_engineering.py           # Training: train model
│       │   ├── inference_stage2.py                 # Prediction: use pre-trained
│       │   └── 02b_description_encoding/
│       │       └── global_cluster_model.pkl        # Pre-trained clustering model
│       │
│       ├── 03_fraud_relabeling/
│       │   ├── 03_fraud_relabeling.py              # Training: complex matching
│       │   └── inference_stage3.py                 # Prediction: Fraud=0
│       │
│       ├── 04_encoding/
│       │   ├── 04_encoding.py                      # Training: encode from by_member
│       │   └── inference_stage4.py                 # Prediction: encode from by_member
│       │
│       ├── run_train_pipeline.py                   # Training automation script
│       ├── run_pred_pipeline.py                    # Prediction automation script
│       ├── training_pipeline.ipynb                 # Training notebook
│       └── prediction_pipeline.ipynb               # Prediction notebook
│
├── Makefile                            # Build automation
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
└── setup.py                            # Setup script
```

---

## 🔄 Data Flow

### Training Pipeline

```
Stage 1: Data Cleaning
  data/train/raw/*.csv
    → data/train/cleaned/*.csv

Stage 2: Feature Engineering (Training)
  data/train/cleaned/*.csv
    → data/train/clustered_out/*.csv (with cluster_id)
    → Saves: global_cluster_model.pkl

Stage 3: Fraud Matching
  data/train/clustered_out/*.csv
    → data/train/by_member/temp/member_*.csv (temporary)
    → data/train/by_member/{matched,unmatched,no_fraud}/member_*.csv
    → Deletes: data/train/by_member/temp/

Stage 4: Feature Encoding
  data/train/by_member/{matched,unmatched,no_fraud}/member_*.csv
    → data/train/final/{matched,unmatched,no_fraud}/member_*.csv
```

### Prediction Pipeline

```
Stage 1: Data Cleaning
  data/pred/raw/*.csv
    → data/pred/cleaned/*.csv

Stage 2: Clustering (Inference)
  data/pred/cleaned/*.csv
    → data/pred/clustered_out/*.csv (with cluster_id)
    → Uses: global_cluster_model.pkl (pre-trained)

Stage 3: Reorganize by Member
  data/pred/clustered_out/*.csv
    → data/pred/by_member/member_*.csv (with Fraud=0)

Stage 4: Feature Encoding
  data/pred/by_member/member_*.csv
    → data/pred/final/member_*.csv
```

---

## 📋 Directory Creation

### Automatic Creation

The following directories are **automatically created** when needed:

**Training Pipeline:**
- ✅ `data/train/by_member/temp/` - Created by `reorganize_by_member()`
- ✅ `data/train/by_member/{matched,unmatched,no_fraud}/` - Created by `process_member_files_for_fraud()`
- ✅ `data/train/final/{matched,unmatched,no_fraud}/` - Created by `encode_features()`

**Prediction Pipeline:**
- ✅ `data/pred/by_member/` - Created by `reorganize_with_fraud_by_member()`
- ✅ `data/pred/final/` - Created by `encode_features()`

### Setup Command

To create all directories at once:

```bash
make setup
# or
python setup.py
```

This creates:
```
data/train/raw/
data/train/cleaned/
data/train/clustered_out/
data/train/by_member/temp/
data/train/by_member/matched/
data/train/by_member/unmatched/
data/train/by_member/no_fraud/
data/train/final/matched/
data/train/final/unmatched/
data/train/final/no_fraud/

data/pred/raw/
data/pred/cleaned/
data/pred/clustered_out/
data/pred/by_member/
data/pred/final/
```

---

## 🗑️ Cleanup

### Clean Training Data

```bash
make clean-train
```

Removes:
- `data/train/cleaned/*`
- `data/train/clustered_out/*`
- `data/train/by_member/*` (including temp, matched, unmatched, no_fraud)
- `data/train/final/*`

### Clean Prediction Data

```bash
make clean-pred
```

Removes:
- `data/pred/cleaned/*`
- `data/pred/clustered_out/*`
- `data/pred/by_member/*`
- `data/pred/final/*`

### Clean All

```bash
make clean-all
# or
make clean
```

Removes both training and prediction data.

---

## 📊 Directory Sizes (Typical)

| Directory | Size (Example) | Description |
|-----------|----------------|-------------|
| `raw/` | 60 MB | Original CSV |
| `cleaned/` | 58 MB | Cleaned, slight reduction |
| `clustered_out/` | 60 MB | +cluster_id column |
| `by_member/temp/` | 60 MB | Reorganized (temporary) |
| `by_member/{matched,unmatched,no_fraud}/` | 60 MB | +Fraud column |
| `final/` | 50 MB | Encoded features only |

**Total Training Pipeline:** ~300 MB (for 580K transactions)

---

## ⚠️ Important Notes

### Temporary Directory

- `data/train/by_member/temp/` is **automatically deleted** after Stage 3 completes
- If Stage 3 fails, `temp/` may remain (for debugging)
- Manually delete with: `rm -rf data/train/by_member/temp/`

### Directory Naming

- **`by_member/`** contains categorized member files (matched/unmatched/no_fraud)
- This is the output of Stage 3 (Fraud Matching)
- Despite the name, it's organized by fraud category, not just member ID

### Prediction vs Training

| Aspect | Training | Prediction |
|--------|----------|------------|
| Stage 2 | Train model | Use pre-trained model |
| Stage 3 | Complex fraud matching | Simple Fraud=0 |
| Output structure | by_member/{matched,unmatched,no_fraud}/ | by_member/ (flat) |
| Temp directory | Uses temp/ | No temp needed |

---

## 🔍 Troubleshooting

### "Directory not found" errors

Run setup first:
```bash
make setup
```

### Temp directory not deleted

Check if Stage 3 completed successfully. If interrupted, manually clean:
```bash
rm -rf data/train/by_member/temp/
```

### Permission errors

Ensure write permissions:
```bash
chmod -R u+w data/
```

---

Generated: 2025-11-19
