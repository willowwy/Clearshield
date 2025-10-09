# src/03_feature_engineering/__init__.py

import os
import pandas as pd
from .user_features import generate_user_features
from .amount_features import generate_amount_features

def _read_raw(path: str) -> pd.DataFrame:
    # safer csv read to avoid dtype warnings
    return pd.read_csv(path, low_memory=False)

def build_and_save_user_features(
    raw_path: str = "01_data_cleaning/raw/transaction_data.csv",
    output_path: str = "01_data_cleaning/processed/03_feature_engineering/user_features.csv",
):
    """
    Extract from raw dataset and save to processed/03_feature_engineering directory.
    """
    df = pd.read_csv(raw_path)
    features = generate_user_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"User 03_feature_engineering successfully saved to {output_path}")

def build_and_save_amount_features(
    raw_path: str = "01_data_cleaning/raw/transaction_data.csv",
    output_path: str = "01_data_cleaning/processed/03_feature_engineering/amount_features.csv",
):
    df = _read_raw(raw_path)
    feats = generate_amount_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feats.to_csv(output_path, index=False)
    print(f"Amount 03_feature_engineering successfully saved to {output_path}")

def build_all_features():
    build_and_save_user_features()
    build_and_save_amount_features()
