"""
Utilities to build text embeddings for transaction descriptions.

The main entry point is `generate_description_embeddings`, which:
  1. Loads a pretrained BERT model (configurable).
  2. Encodes the chosen description column into sentence-level embeddings.
  3. Applies an optional dimensionality reduction step (defaults to PCA).

Example:
    >>> import pandas as pd
    >>> from .description_encoder import generate_description_embeddings
    >>> df = pd.read_csv("transaction_data.csv")
    >>> desc_embeddings = generate_description_embeddings(df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The 'transformers' package is required for description encoding. "
        "Install it via `pip install transformers`."
    ) from exc

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scikit-learn is required for PCA dimensionality reduction. "
        "Install it via `pip install scikit-learn`."
    ) from exc

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


@dataclass(frozen=True)
class DescriptionEncoderConfig:
    """Configuration container for description embedding generation."""

    text_column: str = "Description"
    model_name: str = "bert-base-uncased"
    batch_size: int = 32
    max_length: int = 64
    output_dim: int = 50
    reduction_method: str = "pca"
    random_state: int = 42
    device: Optional[str] = None
    id_columns: Sequence[str] = (
        "Member ID",
        "Account ID",
        "Transaction ID",
        "TxnOrdinal",
    )
    include_text_column: bool = False


def _iter_batches(values: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings with mask awareness."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return summed / lengths


def _reduce_embeddings(
    embeddings: np.ndarray,
    method: Optional[str],
    target_dim: Optional[int],
    random_state: int,
) -> tuple[np.ndarray, Optional[str]]:
    """Apply dimensionality reduction; returns transformed embeddings and a suffix label."""
    if method is None or target_dim is None:
        return embeddings, "bert"

    method = method.lower()
    if method == "none":
        return embeddings, "bert"

    if target_dim >= embeddings.shape[1]:
        # Nothing to reduce, return original representation.
        return embeddings, "bert"

    if method == "pca":
        reducer = PCA(n_components=target_dim, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        return reduced, "pca"

    if method == "umap":
        if not _HAS_UMAP:
            raise ImportError(
                "UMAP reduction requested but `umap-learn` is not installed. "
                "Install it via `pip install umap-learn numba`."
            )
        reducer = UMAP(n_components=target_dim, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        return reduced, "umap"

    raise ValueError(f"Unsupported reduction method '{method}'. Use 'pca', 'umap', or 'none'.")


def _select_identifier_columns(df: pd.DataFrame, candidate_cols: Sequence[str]) -> pd.DataFrame:
    available_cols = [col for col in candidate_cols if col in df.columns]
    if available_cols:
        return df[available_cols].reset_index(drop=True)

    # Fallback: preserve the original positional index
    return pd.DataFrame({"row_index": df.index}).reset_index(drop=True)


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover - macOS specific
        return "mps"
    return "cpu"


def generate_description_embeddings(
    transactions: pd.DataFrame,
    config: DescriptionEncoderConfig | None = None,
) -> pd.DataFrame:
    """
    Encode transaction descriptions into dense vectors.

    Args:
        transactions: Source DataFrame containing at least the `config.text_column`.
        config: Optional DescriptionEncoderConfig. If omitted, defaults are used.

    Returns:
        DataFrame with identifier columns and embedding features.
    """
    cfg = config or DescriptionEncoderConfig()

    if cfg.text_column not in transactions.columns:
        raise KeyError(
            f"Column '{cfg.text_column}' not found in the provided DataFrame. "
            f"Available columns: {list(transactions.columns)}"
        )

    device = _resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name)
    model.to(device)
    model.eval()

    text_series = transactions[cfg.text_column].fillna("").astype(str)
    texts = text_series.tolist()

    all_embeddings = []
    with torch.no_grad():
        for batch_texts in _iter_batches(texts, cfg.batch_size):
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            sentence_embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(sentence_embeddings.cpu().numpy())

    if not all_embeddings:
        raise ValueError("No descriptions were encoded; check that the input DataFrame is not empty.")

    embeddings_matrix = np.vstack(all_embeddings)
    reduced_embeddings, suffix = _reduce_embeddings(
        embeddings_matrix,
        cfg.reduction_method,
        cfg.output_dim,
        cfg.random_state,
    )

    feature_cols = [
        f"description_embedding_{suffix}_{idx:03d}"
        for idx in range(reduced_embeddings.shape[1])
    ]

    feature_df = pd.DataFrame(reduced_embeddings, columns=feature_cols)
    id_df = _select_identifier_columns(transactions, cfg.id_columns)

    output_parts = [id_df, feature_df]
    if cfg.include_text_column:
        output_parts.append(text_series.reset_index(drop=True).to_frame(name=cfg.text_column))

    return pd.concat(output_parts, axis=1)
