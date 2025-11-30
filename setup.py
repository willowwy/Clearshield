#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearShield - Fraud Detection System
Setup script for project initialization
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 9):
        print(f"✗ Python 3.9+ is required. Current version: {sys.version}")
        sys.exit(1)
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")


def create_directories():
    """Create necessary data directories using centralized config"""
    print("Creating project directories...")

    # Import centralized configuration
    try:
        from config.pipeline_config import get_train_config, get_pred_config

        # Create training directories
        print("\n  Training Pipeline Directories:")
        train_config = get_train_config()
        train_config.create_directories(verbose=False)

        # Print created training directories
        for key, path in train_config.paths.items():
            if 'train' in str(path):
                print(f"    ✓ {path.relative_to(train_config.PROJECT_ROOT)}")

        # Create prediction directories
        print("\n  Prediction Pipeline Directories:")
        pred_config = get_pred_config()
        pred_config.create_directories(verbose=False)

        # Print created prediction directories
        for key, path in pred_config.paths.items():
            if 'pred' in str(path):
                print(f"    ✓ {path.relative_to(pred_config.PROJECT_ROOT)}")

    except ImportError as e:
        print(f"  ⚠ Warning: Could not import pipeline_config: {e}")
        print("  Creating basic directory structure instead...")

        # Fallback: create basic directory structure
        basic_dirs = [
            'data/train/raw',
            'data/train/cleaned',
            'data/train/clustered_out',
            'data/train/by_member/temp',
            'data/train/by_member/matched',
            'data/train/by_member/unmatched',
            'data/train/by_member/no_fraud',
            'data/train/final/matched',
            'data/train/final/unmatched',
            'data/train/final/no_fraud',
            'data/pred/raw',
            'data/pred/cleaned',
            'data/pred/clustered_out',
            'data/pred/by_member',
            'data/pred/final',
        ]

        for directory in basic_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"    ✓ {directory}")

    # Create additional project directories
    print("\n  Additional Directories:")
    additional_dirs = [
        'notebooks',
        'docs',
        'config',
        'src/data_preprocess/02_feature_engineering/02b_description_encoding',
    ]

    for directory in additional_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"    ✓ {directory}")

    print("\n✓ All directories created successfully!\n")


def install_requirements():
    """Install Python dependencies from requirements.txt"""
    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        sys.exit(1)


def main():
    """Main setup function"""
    print("=" * 60)
    print("ClearShield - Fraud Detection System")
    print("Project Setup")
    print("=" * 60)
    print()

    check_python_version()
    create_directories()
    install_requirements()

    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("\n📊 Training Pipeline (5 stages):")
    print("1. Place your training data in data/train/raw/")
    print("2. Run: python src/data_preprocess/run_train_pipeline.py")
    print("   Or use: src/data_preprocess/train_pipeline.ipynb")
    print("\n   Pipeline stages:")
    print("   - Stage 1: Data Cleaning")
    print("   - Stage 2: Feature Engineering (BERT + Clustering)")
    print("   - Stage 3: Fraud Matching & Re-labeling")
    print("   - Stage 4: Feature Encoding")
    print("   - Stage 5: Vulnerability Scanning (optional)")
    print("\n🔮 Prediction Pipeline:")
    print("3. Place your prediction data in data/pred/raw/")
    print("4. Run: python src/data_preprocess/run_pred_pipeline.py")
    print("   Or use: src/data_preprocess/pred_pipeline.ipynb")
    print("\n💡 Tip: Use --help flag to see all available options")
    print("   Example: python src/data_preprocess/run_train_pipeline.py --help")
    print("\nFor more details, see README.md")
    print()


if __name__ == "__main__":
    main()
