# src/features/amount_features.py

import pandas as pd
import numpy as np
from typing import Optional

# -----------------------------
# Tunable constants
# -----------------------------
ROLLING_N = 10                 # how many previous txns to look back
HIGH_VALUE_SIGMA = 3.0         # mean + 3*std threshold
CONSEC_WINDOW_MIN = 30         # time window (minutes) for "is_consecutive"
CONSEC_MIN_COUNT = 3           # at least N txns in the window
AMOUNT_TOL = 0.01              # tolerance to match buckets or "similar amounts" (~$0.01)
FRAUD_PRONE_BUCKETS = [50.0, 500.0]  # categorical bucket anchors


def _parse_amount(series: pd.Series) -> pd.Series:
    """Convert currency-like strings to float. e.g., '$55.00 ' -> 55.0"""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def _parse_time_hhmm(col: pd.Series) -> pd.Series:
    """
    Parse time stored as integers like 35, 110, 132 into HH:MM (00:35, 01:10, 01:32).
    """
    s = col.astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
    hh = s.str.slice(0, 2).astype(int)
    mm = s.str.slice(2, 4).astype(int)
    return pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")


def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine 'Post Date' and 'Post Time' into a single pandas.Timestamp."""
    post_date = pd.to_datetime(df["Post Date"], errors="coerce")
    delta = _parse_time_hhmm(df["Post Time"])
    return post_date + delta


def _mark_bucket(amount: pd.Series, anchors=FRAUD_PRONE_BUCKETS, tol=AMOUNT_TOL) -> pd.Series:
    """Map amount into categorical bucket: '50', '500', or 'none'."""
    labels = []
    for v in amount.values:
        if pd.isna(v):
            labels.append("none")
            continue
        hit = "none"
        for a in anchors:
            if abs(v - a) <= tol:
                hit = str(int(a))
                break
        labels.append(hit)
    return pd.Series(labels, index=amount.index, name="Fraud_Prone_Amount_Bucket")


def generate_amount_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build transaction-level amount features.
    Output includes:
      - Member ID, TxnOrdinal, TxnTimestamp, AmountFloat
      - Fraud_Prone_Amount_Bucket
      - is_high_value (global anomaly detection)
      - is_consecutive (short-term repetition)
      - rolling stats (past N): mean/std/max/min
    """
    df = transactions.copy()

    # --- Parse amount and timestamp ---
    df["AmountFloat"] = _parse_amount(df["Amount"])
    df["TxnTimestamp"] = _build_timestamp(df)

    # --- Order transactions per member ---
    df = df.sort_values(["Member ID", "TxnTimestamp"]).reset_index(drop=True)
    df["TxnOrdinal"] = df.groupby("Member ID").cumcount() + 1

    # --- Rolling stats (past N) ---
    g = df.groupby("Member ID", group_keys=False)
    df["mean_amount_pastN"] = g["AmountFloat"].apply(
        lambda s: s.rolling(ROLLING_N, min_periods=1).mean().shift(1)
    )
    df["std_amount_pastN"] = g["AmountFloat"].apply(
        lambda s: s.rolling(ROLLING_N, min_periods=2).std().shift(1)
    )
    df["max_amount_pastN"] = g["AmountFloat"].apply(
        lambda s: s.rolling(ROLLING_N, min_periods=1).max().shift(1)
    )
    df["min_amount_pastN"] = g["AmountFloat"].apply(
        lambda s: s.rolling(ROLLING_N, min_periods=1).min().shift(1)
    )

    # --- High-value flag (expanding mean+std) ---
    expanding_mean = g["AmountFloat"].apply(lambda s: s.expanding(min_periods=2).mean().shift(1))
    expanding_std = g["AmountFloat"].apply(lambda s: s.expanding(min_periods=2).std().shift(1))
    df["is_high_value"] = (df["AmountFloat"] > (expanding_mean + HIGH_VALUE_SIGMA * expanding_std)).fillna(False)

    # --- Fraud-prone buckets ---
    df["Fraud_Prone_Amount_Bucket"] = _mark_bucket(df["AmountFloat"])

    # --- is_consecutive: frequent similar amounts in short window ---
    def _count_in_window(group: pd.DataFrame) -> pd.Series:
        times = group["TxnTimestamp"].values.astype("datetime64[ns]")
        amounts = group["AmountFloat"].values
        counts = np.zeros(len(group), dtype=int)
        win_ns = int(CONSEC_WINDOW_MIN * 60 * 1e9)

        left = 0
        for right in range(len(group)):
            while times[right] - times[left] > np.timedelta64(win_ns, "ns"):
                left += 1
            window_idx = range(left, right)
            if len(window_idx) > 0:
                cnt = np.sum(np.abs(amounts[list(window_idx)] - amounts[right]) <= AMOUNT_TOL)
            else:
                cnt = 0
            counts[right] = cnt
        return pd.Series(counts, index=group.index)

    df["_consec_count"] = g.apply(_count_in_window)
    df["is_consecutive"] = (df["_consec_count"] >= (CONSEC_MIN_COUNT - 1))
    df = df.drop(columns=["_consec_count"])

    # --- Final selection ---
    out_cols = [
        "Member ID",
        "TxnOrdinal",
        "TxnTimestamp",
        "AmountFloat",
        "Fraud_Prone_Amount_Bucket",
        "is_high_value",
        "is_consecutive",
        "mean_amount_pastN",
        "std_amount_pastN",
        "max_amount_pastN",
        "min_amount_pastN",
    ]
    return df[out_cols]
