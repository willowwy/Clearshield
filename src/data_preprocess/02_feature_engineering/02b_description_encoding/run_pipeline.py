#!/usr/bin/env python3
"""
End-to-end CLI for the description clustering workflow.

It scans a root directory for CSV files, encodes transaction descriptions
with BERT-tiny, reduces vectors with PCA, auto-selects the cluster count,
and writes mirrored outputs that include a `cluster_id` column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .description_encoder import (
        DEFAULT_MODEL_NAME,
        DEFAULT_RAW_ROOT,
        DEFAULT_TEXT_COLUMN,
        run_pipeline,
    )
except ImportError:  # Allows running as a stand-alone script
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from description_encoder import (  # type: ignore
        DEFAULT_MODEL_NAME,
        DEFAULT_RAW_ROOT,
        DEFAULT_TEXT_COLUMN,
        run_pipeline,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode transaction descriptions with BERT-tiny, perform PCA, and cluster with MiniBatchKMeans.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing CSV files.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face identifier for the encoder model.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Name of the column to encode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for transformer encoding.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum token length for the tokenizer.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=20,
        help="Number of PCA dimensions to keep.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=10,
        help="Minimum number of clusters to evaluate.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=60,
        help="Maximum number of clusters to evaluate.",
    )
    parser.add_argument(
        "--k-step",
        type=int,
        default=10,
        help="Step size when scanning for the best cluster count.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of rows sampled when estimating the best cluster count.",
    )
    parser.add_argument(
        "--cluster-batch-size",
        type=int,
        default=4096,
        help="MiniBatchKMeans batch size for the final clustering pass.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        args.root,
        model_name=args.model_name,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pca_dim=args.pca_dim,
        min_k=args.min_k,
        max_k=args.max_k,
        k_step=args.k_step,
        sample_size=args.sample_size,
        cluster_batch_size=args.cluster_batch_size,
        random_state=args.random_state,
    )
    print(f"[Done] Wrote {len(outputs)} clustered files.")


if __name__ == "__main__":
    main()
