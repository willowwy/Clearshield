"""
Fast GPU-friendly clustering pipeline for transaction descriptions.

This module powers the CLI in `run_pipeline.py` and can also be imported
programmatically. It scans CSV files, encodes the text column with a
small BERT model, reduces the embeddings via PCA, and clusters them with
MiniBatchKMeans. Each processed CSV receives a new `cluster_id` column
and is written to a mirrored folder structure rooted at `clustered_out`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_TEXT_COLUMN = "Transaction Description"
DEFAULT_RAW_ROOT = "/home/ubuntu/data_unzipped"
DEFAULT_CLUSTER_COUNT = 60

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEXT_COLUMN",
    "DEFAULT_RAW_ROOT",
    "DEFAULT_CLUSTER_COUNT",
    "get_device",
    "mean_pool",
    "encode_texts",
    "reduce_pca",
    "process_csv",
    "run_pipeline",
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _batched(iterable: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield list(iterable[idx : idx + batch_size])


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 64,
) -> np.ndarray:
    device = get_device()
    print(f"[Device] Using {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(_batched(texts, batch_size), desc="Encoding"):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def reduce_pca(
    vectors: np.ndarray,
    dim: int = 20,
    *,
    random_state: int = 42,
) -> np.ndarray:
    print(f"[PCA] Reducing to {dim} dimensions...")
    reducer = PCA(n_components=dim, random_state=random_state)
    return reducer.fit_transform(vectors)


def _resolve_output_path(file_path: str) -> str:
    if "/processed/" in file_path:
        return file_path.replace("/processed/", "/clustered_out/")
    if "/raw/" in file_path:
        return file_path.replace("/raw/", "/clustered_out/")

    path = Path(file_path)
    return str(path.parent / "clustered_out" / path.name)


def process_csv(
    file_path: str,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 64,
    pca_dim: int = 20,
    min_k: int | None = None,  # Deprecated: retained for CLI compatibility
    max_k: int | None = None,  # Deprecated: retained for CLI compatibility
    k_step: int | None = None,  # Deprecated: retained for CLI compatibility
    sample_size: int | None = None,  # Deprecated: retained for CLI compatibility
    cluster_batch_size: int = 4096,
    random_state: int = 42,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
) -> str | None:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"[Skip] Cannot read {file_path}: {exc}")
        return None

    if text_column not in df.columns:
        print(f"[Skip] Missing column '{text_column}' in {file_path}")
        return None

    texts = df[text_column].fillna("").astype(str).tolist()
    if not texts:
        print(f"[Skip] Empty text column in {file_path}")
        return None

    embeddings = encode_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    reduced = reduce_pca(embeddings, dim=pca_dim, random_state=random_state)
    effective_k = cluster_count or DEFAULT_CLUSTER_COUNT

    km = MiniBatchKMeans(
        n_clusters=effective_k,
        random_state=random_state,
        batch_size=cluster_batch_size,
        n_init="auto",
    )
    df["cluster_id"] = km.fit_predict(reduced)

    output_path = _resolve_output_path(file_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")
    return output_path


def run_pipeline(
    raw_root: str = DEFAULT_RAW_ROOT,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    text_column: str = DEFAULT_TEXT_COLUMN,
    batch_size: int = 64,
    max_length: int = 64,
    pca_dim: int = 20,
    min_k: int | None = None,
    max_k: int | None = None,
    k_step: int | None = None,
    sample_size: int | None = None,
    cluster_batch_size: int = 4096,
    random_state: int = 42,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
) -> list[str]:
    all_csvs: list[str] = []
    for root, _, files in os.walk(raw_root):
        for name in files:
            if name.endswith(".csv"):
                all_csvs.append(os.path.join(root, name))

    print(f"[Found] {len(all_csvs)} CSV files total.")
    outputs: list[str] = []

    for csv_path in all_csvs:
        result = process_csv(
            csv_path,
            text_column=text_column,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            pca_dim=pca_dim,
            min_k=min_k,
            max_k=max_k,
            k_step=k_step,
            sample_size=sample_size,
            cluster_batch_size=cluster_batch_size,
            random_state=random_state,
            cluster_count=cluster_count,
        )
        if result:
            outputs.append(result)

    return outputs
