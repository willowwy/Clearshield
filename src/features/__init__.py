# src/features/__init__.py

import os
import pandas as pd
from .user_features import generate_user_features
from .amount_features import generate_amount_features

def _read_raw(path: str) -> pd.DataFrame:
    # safer csv read to avoid dtype warnings
    return pd.read_csv(path, low_memory=False)

def build_and_save_user_features(
    raw_path: str = "data/raw/transaction_data.csv",
    output_path: str = "data/processed/features/user_features.csv",
):
    """
    Extract from raw dataset and save to processed/features directory.
    """
    df = pd.read_csv(raw_path)
    features = generate_user_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"User features successfully saved to {output_path}")

def build_and_save_amount_features(
    raw_path: str = "data/raw/transaction_data.csv",
    output_path: str = "data/processed/features/amount_features.csv",
):
    df = _read_raw(raw_path)
    feats = generate_amount_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feats.to_csv(output_path, index=False)
    print(f"Amount features successfully saved to {output_path}")

def build_all_features():
    build_and_save_user_features()
    build_and_save_amount_features()
