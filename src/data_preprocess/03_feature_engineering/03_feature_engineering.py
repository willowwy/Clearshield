#!/usr/bin/env python3
"""
Feature engineering pipeline entry point.

This module serves as the main entry point for the feature engineering pipeline,
wrapping the description encoding workflow from 03b_description_encoding.
"""

from __future__ import annotations

import importlib.util
import sys
import os
from pathlib import Path

# Configuration - can be modified from external scripts
PROCESSED_DIR = '../../../data/processed'
MODEL_NAME = 'prajjwal1/bert-tiny'
TEXT_COLUMN = 'Transaction Description'
BATCH_SIZE = 64
MAX_LENGTH = 64
PCA_DIM = 20
MIN_K = 10
MAX_K = 60
K_STEP = 10
SAMPLE_SIZE = 10000
CLUSTER_BATCH_SIZE = 4096
RANDOM_STATE = 42


def run_stage3(processed_dir=None, model_name=None, text_column=None,
               batch_size=None, max_length=None, pca_dim=None,
               min_k=None, max_k=None, k_step=None, sample_size=None,
               cluster_batch_size=None, random_state=None, verbose=True):
    """
    Run Stage 3: Description Encoding and Clustering

    This function processes CSV files in the processed directory structure,
    encodes transaction descriptions using BERT, performs PCA dimensionality
    reduction, and clusters the embeddings using MiniBatchKMeans.

    Args:
        processed_dir: Root directory containing processed member files (default: PROCESSED_DIR)
        model_name: Hugging Face model identifier (default: MODEL_NAME)
        text_column: Column name to encode (default: TEXT_COLUMN)
        batch_size: Batch size for encoding (default: BATCH_SIZE)
        max_length: Max token length (default: MAX_LENGTH)
        pca_dim: PCA dimensions (default: PCA_DIM)
        min_k: Minimum cluster count (default: MIN_K)
        max_k: Maximum cluster count (default: MAX_K)
        k_step: Step size for k search (default: K_STEP)
        sample_size: Sample size for k estimation (default: SAMPLE_SIZE)
        cluster_batch_size: MiniBatchKMeans batch size (default: CLUSTER_BATCH_SIZE)
        random_state: Random seed (default: RANDOM_STATE)
        verbose: Print progress information (default: True)

    Returns:
        List of output file paths
    """
    # Use global config if not specified
    processed_dir = processed_dir or PROCESSED_DIR
    model_name = model_name or MODEL_NAME
    text_column = text_column or TEXT_COLUMN
    batch_size = batch_size or BATCH_SIZE
    max_length = max_length or MAX_LENGTH
    pca_dim = pca_dim or PCA_DIM
    min_k = min_k or MIN_K
    max_k = max_k or MAX_K
    k_step = k_step or K_STEP
    sample_size = sample_size or SAMPLE_SIZE
    cluster_batch_size = cluster_batch_size or CLUSTER_BATCH_SIZE
    random_state = random_state or RANDOM_STATE

    if verbose:
        print("=" * 60)
        print("STAGE 3: DESCRIPTION ENCODING AND CLUSTERING")
        print("=" * 60)
        print(f"Input: {processed_dir}")
        print(f"Model: {model_name}")
        print(f"Text Column: {text_column}")
        print(f"PCA Dimensions: {pca_dim}")
        print(f"Cluster Range: {min_k}-{max_k} (step {k_step})")
        print("")

    # Dynamically import the description_encoder module
    module_path = Path(__file__).parent / "03b_description_encoding" / "description_encoder.py"
    spec = importlib.util.spec_from_file_location("description_encoder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    desc_encoder = importlib.util.module_from_spec(spec)
    sys.modules["description_encoder"] = desc_encoder
    spec.loader.exec_module(desc_encoder)

    # Call the run_pipeline function with configuration
    outputs = desc_encoder.run_pipeline(
        raw_root=processed_dir,
        model_name=model_name,
        text_column=text_column,
        batch_size=batch_size,
        max_length=max_length,
        pca_dim=pca_dim,
        min_k=min_k,
        max_k=max_k,
        k_step=k_step,
        sample_size=sample_size,
        cluster_batch_size=cluster_batch_size,
        random_state=random_state,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 3 COMPLETE")
        print("=" * 60)
        print(f"Processed {len(outputs)} files")
        print(f"Output location: {processed_dir.replace('/processed/', '/clustered_out/')}")

    return outputs


def main() -> None:
    """
    Main entry point for the feature engineering pipeline.

    Currently runs the description encoding and clustering pipeline.
    """
    run_stage3(verbose=True)


if __name__ == "__main__":
    main()
