#!/usr/bin/env python3
"""
Stage 2 Inference: Apply pre-trained clustering model to new data
Loads saved PCA + KMeans model and assigns cluster IDs without retraining
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration
DEFAULT_MODEL_PATH = "cluster_model.pkl"
TEXT_COLUMN = "Transaction Description"
MAX_LENGTH = 64
BATCH_SIZE = 512


def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden, attention_mask):
    """Mean pooling for BERT embeddings"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_batch(texts, tokenizer, model, device):
    """Encode a batch of texts using BERT"""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return pooled.cpu().numpy()


def infer_clusters_for_csv(
    input_csv: str,
    output_csv: str,
    model_path: str = DEFAULT_MODEL_PATH,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True
):
    """
    Apply pre-trained clustering model to a single CSV file

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file with cluster_id column
        model_path: Path to saved model pickle file (PCA + KMeans)
        text_column: Name of the text column to encode
        batch_size: Batch size for encoding
        verbose: Print progress information
    """
    if verbose:
        print(f"[Load Model] {model_path}")

    # Load pre-trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    payload = joblib.load(model_path)
    ipca = payload["ipca"]
    kmeans = payload["kmeans"]
    model_name = payload["model_name"]

    if verbose:
        print(f"  BERT Model: {model_name}")
        print(f"  PCA Components: {ipca.n_components}")
        print(f"  KMeans Clusters: {kmeans.n_clusters}")

    # Load input CSV
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in {input_csv}")

    texts = df[text_column].fillna("").astype(str).tolist()
    total = len(texts)

    if verbose:
        print(f"[Process] {input_csv}")
        print(f"  Rows: {total}")

    # Initialize BERT model
    device = get_device()
    if verbose:
        print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Encode texts in batches
    all_embeddings = []
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embeddings = encode_batch(batch, tokenizer, model, device)
        all_embeddings.append(embeddings)

        if verbose:
            processed = min(i + batch_size, total)
            pct = processed / total * 100
            print(f"  Encoding: {processed}/{total} ({pct:.1f}%)", end='\r')

    if verbose:
        print()  # New line after progress

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Apply PCA transformation
    if verbose:
        print("  Applying PCA...")
    reduced = ipca.transform(embeddings).astype(np.float32)

    # Assign clusters (manual nearest-center for consistency)
    if verbose:
        print("  Assigning clusters...")
    centers = kmeans.cluster_centers_.astype(np.float32)

    # Compute distances to all centers
    diff = reduced[:, None, :] - centers[None, :, :]
    dists = (diff ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)

    # Add cluster_id column
    df["cluster_id"] = labels

    # Save output
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"[Done] Saved → {output_csv}")
        print(f"  Cluster distribution: min={labels.min()}, max={labels.max()}, unique={len(np.unique(labels))}")

    return labels


def infer_clusters_for_directory(
    input_dir: str,
    output_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True
):
    """
    Apply pre-trained clustering model to all CSV files in a directory

    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory to save output CSV files
        model_path: Path to saved model pickle file
        text_column: Name of the text column to encode
        batch_size: Batch size for encoding
        verbose: Print progress information

    Returns:
        List of output file paths
    """
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return []

    if verbose:
        print(f"[Inference] Found {len(csv_files)} CSV files")

    output_files = []
    for i, csv_file in enumerate(csv_files, 1):
        if verbose:
            print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")

        # Mirror directory structure
        rel_path = csv_file.relative_to(input_path)
        output_csv = Path(output_dir) / rel_path

        try:
            infer_clusters_for_csv(
                str(csv_file),
                str(output_csv),
                model_path=model_path,
                text_column=text_column,
                batch_size=batch_size,
                verbose=verbose
            )
            output_files.append(str(output_csv))
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            continue

    if verbose:
        print(f"\n[Complete] Processed {len(output_files)}/{len(csv_files)} files")

    return output_files


def main():
    """Command-line interface for inference"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply pre-trained clustering model to new data"
    )
    parser.add_argument("--input", required=True, help="Input CSV file or directory")
    parser.add_argument("--output", required=True, help="Output CSV file or directory")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to cluster_model.pkl"
    )
    parser.add_argument(
        "--text-column",
        default=TEXT_COLUMN,
        help="Name of text column"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for encoding"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file inference
        infer_clusters_for_csv(
            args.input,
            args.output,
            model_path=args.model,
            text_column=args.text_column,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
    elif input_path.is_dir():
        # Directory inference
        infer_clusters_for_directory(
            args.input,
            args.output,
            model_path=args.model,
            text_column=args.text_column,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
