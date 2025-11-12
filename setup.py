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
    """Create necessary data directories"""
    directories = ['data/raw']

    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

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
    print("1. Place your raw CSV files in data/raw/")
    print("2. Open src/data_preprocess/pipeline.ipynb to run preprocessing")
    print()


if __name__ == "__main__":
    main()
