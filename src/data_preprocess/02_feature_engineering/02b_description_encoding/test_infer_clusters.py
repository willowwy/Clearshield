#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

TEXT_COL = "Transaction Description"
MAX_LENGTH = 64
BATCH_SIZE = 512


# ------------------------------------------
# Helpers
# ------------------------------------------

def device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_batch(texts, tok, mdl, dev):
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(dev)
    with torch.no_grad():
        out = mdl(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        return pooled.cpu().numpy()


# ------------------------------------------
# Main Inference
# ------------------------------------------

def infer_one_csv(input_csv, output_csv, model_pkl):

    print(f"[Model] Loading: {model_pkl}")
    payload = joblib.load(model_pkl)

    ipca = payload["ipca"]
    kmeans = payload["kmeans"]
    model_name = payload["model_name"]

    df = pd.read_csv(input_csv)
    if TEXT_COL not in df.columns:
        raise KeyError(f"Column '{TEXT_COL}' not found in input CSV")

    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    total = len(texts)
    print(f"[Infer] Rows: {total}")

    # ---- Load BERT ----
    dev = device()
    print(f"[Infer] Using device: {dev}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()

    # ---- Encode ----
    all_emb = []
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb = encode_batch(batch, tok, mdl, dev)
        all_emb.append(emb)
        processed = min(i + BATCH_SIZE, total)
        pct = processed / total * 100
        print(f"[Infer][Encoding] {processed}/{total} ({pct:.1f}%)")

    embeddings = np.vstack(all_emb).astype(np.float32)

    # ---- PCA ----
    print("[Infer] PCA.transform...")
    reduced = ipca.transform(embeddings).astype(np.float32)

    # ---- Manual nearest-center assignment ----
    print("[Infer] Manual nearest-center assignment...")
    centers = kmeans.cluster_centers_.astype(np.float32)  # (K, D)

    diff = reduced[:, None, :] - centers[None, :, :]
    dists = (diff ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)

    # ---- Write ----
    df["cluster_id"] = labels
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Done] Saved → {output_csv}")


# ------------------------------------------
# CLI
# ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--model",
        default="/home/ubuntu/global_cluster_model.pkl",
        help="Path to global_cluster_model.pkl",
    )
    args = ap.parse_args()

    infer_one_csv(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
