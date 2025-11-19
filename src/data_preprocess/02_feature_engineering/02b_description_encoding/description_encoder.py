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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_COLUMN = "Transaction Description"

try:
    from .run_pipeline import DEFAULT_RAW_ROOT, DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - fallback when run as a script
    try:
        from run_pipeline import DEFAULT_RAW_ROOT, DEFAULT_OUTPUT_ROOT  # type: ignore
    except ImportError:
        DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "cleaned"
        DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "clustered_out"
DEFAULT_CLUSTER_COUNT = 60

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEXT_COLUMN",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RAW_ROOT",
    "DEFAULT_CLUSTER_COUNT",
    "get_device",
    "mean_pool",
    "encode_texts",
    "reduce_pca",
    "process_csv",
    "run_pipeline",
]


def _log(message: str, *, verbose: bool = False, always: bool = False) -> None:
    if always or verbose:
        print(message)


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
    verbose: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    device = get_device()
    _log(f"[Device] Using {device}", verbose=verbose)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(device).eval()
    except OSError as exc:
        raise RuntimeError(
            f"Model '{model_name}' does not provide safetensors weights. "
            "Select a safetensors-enabled model or upgrade torch to >=2.6."
        ) from exc

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        iterator: Iterable[list[str]] = _batched(texts, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", leave=False)
        for batch in iterator:
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
    verbose: bool = True,
) -> np.ndarray:
    n_samples, n_features = vectors.shape
    effective_dim = min(dim, n_samples, n_features)
    if effective_dim < dim:
        _log(
            f"[Adjust][PCA] Requested {dim} dimensions but only "
            f"{n_samples} samples and {n_features} features; "
            f"using {effective_dim} components.",
            verbose=verbose,
        )
    if effective_dim < 1:
        raise ValueError("Cannot perform PCA with fewer than 1 component.")

    _log(f"[PCA] Reducing to {effective_dim} dimensions...", verbose=verbose)
    reducer = PCA(n_components=effective_dim, random_state=random_state)
    return reducer.fit_transform(vectors)


def _resolve_output_path(
    file_path: str | Path,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    source = Path(file_path)
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    try:
        relative = source.relative_to(raw_root_path)
    except ValueError:
        relative = source.name
    return output_root_path / relative


def process_csv(
    file_path: str,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 64,
    pca_dim: int = 20,
    cluster_batch_size: int = 4096,
    random_state: int = 42,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    verbose: bool = True,
    show_progress: bool = True,
) -> str | None:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        _log(f"[Skip] Cannot read {file_path}: {exc}", always=True)
        return None

    if text_column not in df.columns:
        _log(f"[Skip] Missing column '{text_column}' in {file_path}", always=True)
        return None

    texts = df[text_column].fillna("").astype(str).tolist()
    if not texts:
        _log(f"[Skip] Empty text column in {file_path}", always=True)
        return None

    embeddings = encode_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        verbose=verbose,
        show_progress=show_progress,
    )
    reduced = reduce_pca(
        embeddings,
        dim=pca_dim,
        random_state=random_state,
        verbose=verbose,
    )
    n_samples = len(reduced)
    effective_k = cluster_count or DEFAULT_CLUSTER_COUNT
    if n_samples < effective_k:
        effective_k = max(1, n_samples)

    km = MiniBatchKMeans(
        n_clusters=effective_k,
        random_state=random_state,
        batch_size=cluster_batch_size,
        n_init="auto",
    )
    df["cluster_id"] = km.fit_predict(reduced)

    output_path = _resolve_output_path(
        file_path,
        raw_root=raw_root,
        output_root=output_root,
    )
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    _log(f"[Saved] {output_path}", verbose=verbose)
    return str(output_path)


def run_pipeline(
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    text_column: str = DEFAULT_TEXT_COLUMN,
    batch_size: int = 64,
    max_length: int = 64,
    pca_dim: int = 20,
    cluster_batch_size: int = 4096,
    random_state: int = 42,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    verbose: bool = False,
    show_progress: bool = True,
) -> list[str]:
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    all_csvs = sorted(str(path) for path in raw_root_path.rglob("*.csv"))
    _log(
        f"[Scan] Found {len(all_csvs)} CSV file(s) in {raw_root_path}",
        always=True,
    )
    outputs: list[str] = []

    for csv_path in all_csvs:
        result = process_csv(
            csv_path,
            text_column=text_column,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            pca_dim=pca_dim,
            cluster_batch_size=cluster_batch_size,
            random_state=random_state,
            cluster_count=cluster_count,
            raw_root=raw_root_path,
            output_root=output_root_path,
            verbose=verbose,
            show_progress=show_progress,
        )
        if result:
            outputs.append(result)

    _log(
        f"[Done] Saved {len(outputs)} clustered file(s) to {output_root_path}",
        always=True,
    )
    return outputs
