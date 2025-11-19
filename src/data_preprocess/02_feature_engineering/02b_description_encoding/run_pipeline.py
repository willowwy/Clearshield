#!/usr/bin/env python3
"""
Global clustering pipeline for transaction descriptions.

This script aggregates every CSV under `data/cleaned`, encodes the
transaction descriptions once, discovers a global cluster count, and
writes clustered files plus a short summary to `data/clustered_out`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "cleaned"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "clustered_out"

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

try:  # Prefer relative imports when executed as a module.
    from .description_encoder import (
        DEFAULT_MODEL_NAME,
        DEFAULT_TEXT_COLUMN,
        encode_texts,
        reduce_pca,
    )
except ImportError:  # pragma: no cover - fallback for `python run_pipeline.py`
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from description_encoder import (  # type: ignore
        DEFAULT_MODEL_NAME,
        DEFAULT_TEXT_COLUMN,
        encode_texts,
        reduce_pca,
    )


def heuristic_k(
    vectors: np.ndarray,
    min_k: int = 10,
    max_k: int = 60,
    step: int = 10,
    *,
    sample_size: int = 10_000,
    random_state: int = 42,
    verbose: bool = False,
) -> int:
    """Pick a cluster count by minimizing inertia over a sampled subset."""

    if len(vectors) == 0:
        raise ValueError("Cannot estimate clusters with no vectors.")

    rng = np.random.default_rng(random_state)
    subset_size = min(sample_size, len(vectors))
    sample_idx = rng.choice(len(vectors), subset_size, replace=False)
    sample = vectors[sample_idx]

    scores: list[tuple[int, float]] = []
    for k in range(min_k, max_k + 1, step):
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=2048,
            n_init="auto",
        )
        km.fit(sample)
        scores.append((k, km.inertia_))

    best_k = min(scores, key=lambda item: item[1])[0]
    if verbose:
        print(f"[Auto-K] Range {min_k}-{max_k}, selected k={best_k}")
    return best_k


def _iter_csv_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def run_pipeline(
    input_root: str | Path = DEFAULT_RAW_ROOT,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    text_column: str = DEFAULT_TEXT_COLUMN,
    batch_size: int = 64,
    max_length: int = 64,
    pca_dim: int = 20,
    min_k: int = 10,
    max_k: int = 60,
    k_step: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> None:
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)
    csv_files = _iter_csv_files(input_root_path)
    if not csv_files:
        print(f"[Abort] No CSV files under {input_root_path}")
        return

    print(
        f"[Start] Clustering descriptions from {len(csv_files)} file(s) "
        f"in {input_root_path}"
    )

    all_texts: list[str] = []
    file_lengths: list[int] = []
    valid_files: list[Path] = []
    skipped: list[str] = []

    def _note_skip(message: str) -> None:
        skipped.append(message)
        print(f"[Skip] {message}")

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            _note_skip(f"{path}: read error ({exc})")
            continue

        if text_column not in df.columns:
            _note_skip(f"{path}: missing '{text_column}' column")
            continue

        texts = df[text_column].fillna("").astype(str).tolist()
        if not texts:
            _note_skip(f"{path}: empty '{text_column}' column")
            continue

        all_texts.extend(texts)
        file_lengths.append(len(texts))
        valid_files.append(path)

    if not all_texts:
        print("[Abort] No valid descriptions found.")
        if skipped:
            print(f"[Info] Skipped {len(skipped)} file(s).")
        return

    print(
        f"[Info] Encoding {len(all_texts)} descriptions from "
        f"{len(valid_files)} file(s)..."
    )
    embeddings = encode_texts(
        all_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        verbose=verbose,
    )
    reduced = reduce_pca(
        embeddings,
        dim=pca_dim,
        random_state=random_state,
        verbose=verbose,
    )
    best_k = heuristic_k(
        reduced,
        min_k=min_k,
        max_k=max_k,
        step=k_step,
        random_state=random_state,
        verbose=verbose,
    )
    km = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=random_state,
        batch_size=4096,
        n_init="auto",
    )
    cluster_ids = km.fit_predict(reduced)

    offset = 0
    cluster_summary_records: list[dict[str, str | int]] = []
    for path, length in zip(valid_files, file_lengths, strict=True):
        df = pd.read_csv(path).head(length).copy()
        df["cluster_id"] = cluster_ids[offset : offset + length]
        offset += length

        relative_path = path.relative_to(input_root_path)
        out_path = output_root_path / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        for cid, group in df.groupby("cluster_id"):
            sample_texts = group[text_column].head(3).tolist()
            cluster_summary_records.append(
                {
                    "cluster_id": int(cid),
                    "file": relative_path.as_posix(),
                    "count_in_file": int(len(group)),
                    "sample_descriptions": " | ".join(sample_texts),
                }
            )

    summary_df = pd.DataFrame(cluster_summary_records)
    summary_global = (
        summary_df.groupby("cluster_id")
        .agg(
            count_in_file=("count_in_file", "sum"),
            sample_descriptions=("sample_descriptions", lambda x: " | ".join(list(x)[:5])),
        )
        .reset_index()
        .sort_values("count_in_file", ascending=False)
    )
    output_root_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_root_path / "cluster_summary.csv"
    summary_global.to_csv(summary_path, index=False)

    print(
        f"[Done] Saved {len(valid_files)} clustered file(s) "
        f"to {output_root_path}. Summary: {summary_path}"
    )
    if skipped:
        print(f"[Info] Skipped {len(skipped)} file(s); see log above for reasons.")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
