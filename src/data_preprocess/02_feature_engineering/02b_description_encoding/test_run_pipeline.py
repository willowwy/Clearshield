#!/usr/bin/env python3
"""
Global Clustering (single K for entire dataset) with low RAM.
Steps:
1) Scan all CSV recursively, index row ranges.
2) Encode all descriptions to memmap (N x 128).
3) IncrementalPCA -> memmap (N x 20).
4) Choose ONE global K on a random sample, fit MiniBatchKMeans on ALL,
   predict per-file ranges, write each CSV with new 'cluster_id' column.
Outputs:
- /home/ubuntu/embeddings.mmap
- /home/ubuntu/embeddings_20d.mmap
- /home/ubuntu/clustered_out/ (mirrors input tree)
- /home/ubuntu/global_cluster_model.pkl
"""
import os, sys, json, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModel
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_SAFE_MODEL_FOR_TORCH_LOAD"] = "1"

RAW_ROOT = "/home/ubuntu/data_unzipped"
OUT_ROOT = "/home/ubuntu/clustered_out"
MODEL_NAME = "prajjwal1/bert-tiny"
CSV_COL = "Transaction Description"

ENC_HIDDEN_DIM = 128
BATCH_ENCODE = 512
REDUCED_DIM = 20
PCA_BATCH = 20000
KM_BATCH = 20000
SAMPLE_FOR_K = 100_000
GLOBAL_K_MIN, GLOBAL_K_MAX, GLOBAL_K_STEP = 10, 60, 10

MEMMAP_EMB  = "/home/ubuntu/embeddings.mmap"
MEMMAP_20D  = "/home/ubuntu/embeddings_20d.mmap"
INDEX_JSON  = "/home/ubuntu/global_index.json"
MODEL_PKL   = "/home/ubuntu/global_cluster_model.pkl"  # 新增：模型输出路径

def device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def is_garbage(p: Path) -> bool:
    n = p.name
    return n.startswith("._") or "__MACOSX" in p.parts

def list_csvs(root: str):
    return sorted([p for p in Path(root).rglob("*.csv") if not is_garbage(p)])

def log(x): print(x, flush=True)

def build_index(csvs):
    total, spans = 0, []
    for i, p in enumerate(csvs):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            log(f"[Skip-Read] {p} ({e})"); continue
        if CSV_COL not in df.columns:
            log(f"[Skip-Col]  {p} (no '{CSV_COL}')"); continue
        n = len(df)
        if n <= 0:
            log(f"[Skip-Empty] {p}"); continue
        spans.append({"path": str(p), "start": total, "end": total+n, "rows": n})
        total += n
        if (i+1) % 500 == 0: log(f"[Index] {i+1}/{len(csvs)} files, rows={total}")
    return total, spans

def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts

def encode_batch(texts, tok, mdl, dev):
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = mdl(**enc)
        return mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()

def encode_all_to_memmap(spans):
    dev = device()
    log(f"[Stage] Encoding -> {MEMMAP_EMB}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(dev).eval()
    N = spans[-1]["end"]
    mm = np.memmap(MEMMAP_EMB, mode="w+", dtype="float32", shape=(N, ENC_HIDDEN_DIM))
    filled = 0
    for idx, m in enumerate(spans):
        p, s, e, n = m["path"], m["start"], m["end"], m["rows"]
        try:
            df = pd.read_csv(p)
        except Exception as ex:
            log(f"[Skip-Read2] {p} ({ex})"); continue
        texts = df[CSV_COL].fillna("").astype(str).tolist()
        for i in range(0, n, BATCH_ENCODE):
            batch = texts[i:i+BATCH_ENCODE]
            mm[s+i:s+i+len(batch)] = encode_batch(batch, tok, mdl, dev)
        filled += n
        if (idx+1) % 200 == 0: log(f"[Encode] files={idx+1}/{len(spans)}, rows={filled}/{N}")
    del mm; gc.collect()
    log("[Stage] Encoding done.")

def ipca_fit(N):
    log("[Stage] IPCA fit (20D)")
    ipca = IncrementalPCA(n_components=REDUCED_DIM, batch_size=PCA_BATCH)
    mm = np.memmap(MEMMAP_EMB, mode="r", dtype="float32", shape=(N, ENC_HIDDEN_DIM))
    for i in range(0, N, PCA_BATCH):
        ipca.partial_fit(mm[i:i+PCA_BATCH])
        if (i // PCA_BATCH) % 50 == 0: log(f"[IPCA-fit] {min(N, i+PCA_BATCH)}/{N}")
    del mm; gc.collect()
    return ipca

def ipca_transform(N, ipca):
    log(f"[Stage] IPCA transform -> {MEMMAP_20D}")
    mm_in = np.memmap(MEMMAP_EMB, mode="r", dtype="float32", shape=(N, ENC_HIDDEN_DIM))
    mm_out = np.memmap(MEMMAP_20D, mode="w+", dtype="float32", shape=(N, REDUCED_DIM))
    for i in range(0, N, PCA_BATCH):
        mm_out[i:i+PCA_BATCH] = ipca.transform(mm_in[i:i+PCA_BATCH])
        if (i // PCA_BATCH) % 50 == 0: log(f"[IPCA-xform] {min(N, i+PCA_BATCH)}/{N}")
    del mm_in, mm_out; gc.collect()
    log("[Stage] IPCA transform done.")

def choose_global_k(N):
    rng = np.random.default_rng(42)
    sample_n = min(SAMPLE_FOR_K, N)
    idx = rng.choice(N, size=sample_n, replace=False)
    mm = np.memmap(MEMMAP_20D, mode="r", dtype="float32", shape=(N, REDUCED_DIM))
    sample = mm[idx]; del mm
    best_k, best_inertia = None, float("inf")
    log(f"[Auto-K] sample={sample_n}, K {GLOBAL_K_MIN}-{GLOBAL_K_MAX} step {GLOBAL_K_STEP}")
    for k in range(GLOBAL_K_MIN, GLOBAL_K_MAX+1, GLOBAL_K_STEP):
        k_eff = min(k, len(sample))
        km = MiniBatchKMeans(n_clusters=k_eff, random_state=42, batch_size=4096, n_init="auto")
        km.fit(sample)
        if km.inertia_ < best_inertia:
            best_inertia, best_k = km.inertia_, k_eff
        log(f"[Auto-K] K={k_eff} inertia={km.inertia_:.4f}")
    log(f"[Auto-K] BEST_K={best_k}")
    return best_k

def fit_global_kmeans(N, K):
    log(f"[KMeans] Fitting MiniBatchKMeans (K={K})")
    km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, n_init="auto")
    mm = np.memmap(MEMMAP_20D, mode="r", dtype="float32", shape=(N, REDUCED_DIM))
    for i in range(0, N, KM_BATCH):
        km.partial_fit(mm[i:i+KM_BATCH])
        if (i // KM_BATCH) % 50 == 0: log(f"[KMeans-fit] {min(N, i+KM_BATCH)}/{N}")
    del mm; gc.collect()
    return km

def ensure_out_path(in_path: str):
    rel = str(Path(in_path).relative_to(RAW_ROOT))
    out_path = str(Path(OUT_ROOT) / rel)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return out_path

def predict_and_write(spans, N, km):
    log("[Write] Predicting per-file and writing...")
    mm = np.memmap(MEMMAP_20D, mode="r", dtype="float32", shape=(N, REDUCED_DIM))
    for i, m in enumerate(spans):
        p, s, e = m["path"], m["start"], m["end"]
        try:
            df = pd.read_csv(p)
        except Exception as ex:
            log(f"[Skip-Write] {p} ({ex})"); continue
        labels = km.predict(mm[s:e])
        df["cluster_id"] = labels
        outp = ensure_out_path(p)
        df.to_csv(outp, index=False)
        if (i+1) % 200 == 0: log(f"[Write] {i+1}/{len(spans)}")
    del mm; gc.collect()
    log("[Write] Done.")

def main():
    start = time.time()
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    csvs = list_csvs(RAW_ROOT)
    if not csvs:
        log(f"[Error] No CSV under {RAW_ROOT}"); sys.exit(1)
    total_rows, spans = build_index(csvs)
    if total_rows <= 1:
        log("[Error] Not enough rows."); sys.exit(1)
    with open(INDEX_JSON, "w") as f: json.dump({"total_rows": total_rows, "spans": spans}, f)
    log(f"[Index] files={len(spans)}, total_rows={total_rows}")

    if not Path(MEMMAP_EMB).exists():
        encode_all_to_memmap(spans)
    else:
        log("[Stage] Reuse embeddings.mmap")

    if not Path(MEMMAP_20D).exists():
        ipca = ipca_fit(total_rows)
        ipca_transform(total_rows, ipca)
    else:
        log("[Stage] Reuse embeddings_20d.mmap")
        ipca = ipca_fit(total_rows)

    K = choose_global_k(total_rows)
    km = fit_global_kmeans(total_rows, K)
    predict_and_write(spans, total_rows, km)

    joblib.dump({"ipca": ipca, "kmeans": km, "model_name": MODEL_NAME}, MODEL_PKL)
    log(f"[Model] Saved model to {MODEL_PKL}")

    log(f"[Done] Finished in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
