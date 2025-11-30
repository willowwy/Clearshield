# Installation Guide

This guide provides step-by-step instructions for installing and setting up the Clearshield fraud detection system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Method 1: Quick Setup (Recommended)](#method-1-quick-setup-recommended)
  - [Method 2: Manual Setup](#method-2-manual-setup)
- [Post-Installation Setup](#post-installation-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Prerequisites

Before installing Clearshield, ensure your system meets the following requirements:

### System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended)
- **Python**: Version 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended for large datasets)
- **Storage**: At least 5GB free disk space
- **GPU**: Optional but recommended for faster model training (CUDA-compatible GPU)

### Required Software

1. **Python 3.9+**
   - Check your Python version:
     ```bash
     python --version
     # or
     python3 --version
     ```
   - If Python is not installed, download it from [python.org](https://www.python.org/downloads/)

2. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

3. **pip** (Python package installer)
   ```bash
   pip --version
   # or
   python -m pip --version
   ```

---

## Installation Methods

### Method 1: Quick Setup (Recommended)

This method uses the automated setup script to configure the entire environment.

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Clearshield
```

#### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Step 3: Run Setup Script

```bash
python setup.py
```

This script will:
- Verify Python version compatibility (3.9+)
- Create all necessary data directories
- Install all required dependencies from `requirements.txt`
- Display next steps for running the pipelines

#### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, torch, transformers; print('All dependencies installed successfully!')"
```

---

### Method 2: Manual Setup

If you prefer to set up the environment manually or encounter issues with the automated setup.

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Clearshield
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages include:**
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `torch>=2.2.0` - Deep learning framework
- `transformers>=4.38.0` - BERT models for text encoding
- `tqdm>=4.65.0` - Progress bars
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Data visualization
- `seaborn>=0.12.0` - Statistical data visualization
- `jupyter>=1.0.0` - Jupyter notebook support

#### Step 4: Create Directory Structure

**Option A: Using Python**
```bash
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/train/raw', 'data/train/cleaned', 'data/train/clustered_out', 'data/train/by_member/matched', 'data/train/by_member/unmatched', 'data/train/by_member/no_fraud', 'data/train/final/matched', 'data/train/final/unmatched', 'data/train/final/no_fraud', 'data/pred/raw', 'data/pred/cleaned', 'data/pred/clustered_out', 'data/pred/by_member', 'data/pred/final', 'notebooks', 'docs', 'config']]"
```

**Option B: Manual Directory Creation**

Create the following directory structure:
```
data/
├── train/
│   ├── raw/
│   ├── cleaned/
│   ├── clustered_out/
│   ├── by_member/
│   │   ├── matched/
│   │   ├── unmatched/
│   │   └── no_fraud/
│   └── final/
│       ├── matched/
│       ├── unmatched/
│       └── no_fraud/
└── pred/
    ├── raw/
    ├── cleaned/
    ├── clustered_out/
    ├── by_member/
    └── final/
```

---

## Post-Installation Setup

### GPU Support (Optional)

If you have a CUDA-compatible GPU and want to leverage GPU acceleration:

1. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation guide for your operating system

2. **Install PyTorch with CUDA Support**
   ```bash
   # Replace cu118 with your CUDA version (e.g., cu121 for CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU Detection**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
   ```

### Jupyter Notebook Setup (Optional)

If you plan to use the interactive pipeline notebooks:

```bash
# Already installed via requirements.txt, but you can also install the kernel
python -m ipykernel install --user --name=clearshield --display-name "Clearshield (Python 3.9)"
```

To launch Jupyter:
```bash
jupyter notebook
# Navigate to src/data_preprocess/train_pipeline.ipynb
```

---

## Verification

After installation, verify that everything is set up correctly:

### 1. Check Python Environment

```bash
# Ensure virtual environment is activated
which python  # Should point to .venv/bin/python

# Check Python version
python --version  # Should be 3.9 or higher
```

### 2. Verify Dependencies

```bash
# Run a quick dependency check
python -c "
import sys
import pandas as pd
import numpy as np
import sklearn
import torch
import transformers

print(f'Python: {sys.version}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print('All dependencies verified successfully!')
"
```

### 3. Verify Directory Structure

```bash
# Check if key directories exist
ls -la data/train/
ls -la data/pred/
```

### 4. Run Quick Test

```bash
# Test basic functionality
python quick_test.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Python Version Error

**Error:**
```
✗ Python 3.9+ is required. Current version: 3.8.x
```

**Solution:**
- Upgrade Python to version 3.9 or higher
- Use [pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions:
  ```bash
  pyenv install 3.9.0
  pyenv local 3.9.0
  ```

#### Issue 2: pip Installation Failures

**Error:**
```
ERROR: Could not find a version that satisfies the requirement <package>
```

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Try installing with --no-cache-dir
pip install --no-cache-dir -r requirements.txt

# For specific package issues, install individually
pip install pandas numpy scikit-learn
```

#### Issue 3: CUDA/GPU Issues

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in feature engineering scripts
- Use CPU-only mode by setting environment variable:
  ```bash
  export CUDA_VISIBLE_DEVICES=""
  ```

#### Issue 4: Transformers Model Download Failure

**Error:**
```
OSError: Can't load model 'prajjwal1/bert-tiny'
```

**Solution:**
```bash
# Pre-download the BERT model
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('prajjwal1/bert-tiny'); AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')"
```

#### Issue 5: Permission Denied Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Ensure you have write permissions
chmod -R u+w data/

# On Windows, run as administrator or check folder permissions
```

#### Issue 6: Import Errors After Installation

**Error:**
```
ModuleNotFoundError: No module named '<module>'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for error messages in the terminal output
2. **Search existing issues**: Check the project's issue tracker
3. **Create a new issue**: Provide:
   - Your operating system and Python version
   - Full error message and stack trace
   - Steps to reproduce the issue
   - Output of `pip list` showing installed packages

---

## Next Steps

Once installation is complete, you can proceed with:

### 1. Training Pipeline

Place your training data in `data/train/raw/` and run:

```bash
cd src/data_preprocess
python run_train_pipeline.py
```

See the [README.md](../README.md) for detailed pipeline documentation.

### 2. Prediction Pipeline

Place prediction data in `data/pred/raw/` and run:

```bash
cd src/data_preprocess
python run_pred_pipeline.py
```

### 3. Explore Notebooks

For interactive exploration:

```bash
jupyter notebook
# Open: src/data_preprocess/train_pipeline.ipynb
```

### 4. Read Additional Documentation

- [README.md](../README.md) - Project overview and pipeline details
- [features_overview.md](features_overview.md) - Feature engineering details
- [DIRECTORY_STRUCTURE.md](../DIRECTORY_STRUCTURE.md) - Detailed directory structure

---

## Alternative Installation with Makefile

If your system has `make` installed, you can use the Makefile for simplified setup:

```bash
# View available commands
make help

# Complete setup
make setup

# Clean environment
make clean
```

---

## Updating the Installation

To update Clearshield to the latest version:

```bash
# Activate virtual environment
source .venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Re-run setup if needed
python setup.py
```

---

**Last Updated:** November 29, 2024
**Version:** 1.0.0
