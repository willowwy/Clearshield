# src/03_feature_engineering/03b_description_encoding/__init__.py

import os
from typing import Any

import pandas as pd

from .description_encoder import DescriptionEncoderConfig, generate_description_embeddings

DEFAULT_RAW_PATH = "01_data_cleaning/raw/transaction_data.csv"
DEFAULT_OUTPUT_PATH = "01_data_cleaning/processed/03_feature_engineering/description_embeddings.csv"

__all__ = [
    "DescriptionEncoderConfig",
    "generate_description_embeddings",
    "build_and_save_description_embeddings",
    "build_all_features",
]


def _read_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _resolve_config(config: DescriptionEncoderConfig | None, overrides: dict[str, Any]) -> DescriptionEncoderConfig:
    if config is not None and overrides:
        raise ValueError("Pass either `config` or keyword overrides, but not both.")
    if config is not None:
        return config
    return DescriptionEncoderConfig(**overrides)


def build_and_save_description_embeddings(
    raw_path: str = DEFAULT_RAW_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    config: DescriptionEncoderConfig | None = None,
    **config_overrides: Any,
) -> pd.DataFrame:
    """
    Load the transaction dataset, create description embeddings, and persist them.

    Keyword arguments supplied alongside `config` are forwarded to `DescriptionEncoderConfig`.
    """
    cfg = _resolve_config(config, config_overrides)
    df = _read_raw(raw_path)
    embeddings_df = generate_description_embeddings(df, config=cfg)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    embeddings_df.to_csv(output_path, index=False)
    print(f"Description embeddings saved to {output_path}")

    return embeddings_df


def build_all_features() -> pd.DataFrame:
    """Convenience wrapper to mirror the previous module interface."""
    return build_and_save_description_embeddings()
