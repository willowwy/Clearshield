import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


def _select_csv_files(matched_dir: str, mode: str, train_val_test: tuple[float, float, float], seed: int) -> list[str]:
    csv_files = glob.glob(os.path.join(matched_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{matched_dir}'.")

    # deterministic split by file list with seed
    csv_files = sorted(csv_files)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(csv_files))
    rng.shuffle(indices)

    n = len(csv_files)
    train_ratio, val_ratio, test_ratio = train_val_test
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_val_test ratios must sum to 1.0")
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if mode == "train":
        chosen = train_idx
    elif mode == "val":
        chosen = val_idx
    elif mode == "test":
        chosen = test_idx
    else:
        raise ValueError("mode must be one of {'train','val','test'}")

    return [csv_files[i] for i in chosen]


def load_matched_dataframe(matched_dir: str = "matched", mode: str = "train", split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> pd.DataFrame:
    chosen_files = _select_csv_files(matched_dir, mode, split, seed)
    print(f"Loading {len(chosen_files)} files for {mode} mode")
    frames = []
    for path in chosen_files:
        df_part = pd.read_csv(path)
        frames.append(df_part)

    df = pd.concat(frames, axis=0, ignore_index=True)

    # Basic parsing
    if "Post Date" in df.columns:
        df["Post Date"] = pd.to_datetime(df["Post Date"], errors="coerce")
    if "Post Time" in df.columns:
        df["Post Time"] = pd.to_numeric(df["Post Time"], errors="coerce").fillna(0).astype(int)
    if "Amount" in df.columns:
        # Remove currency symbols if present
        df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace('[\$,]', '', regex=True), errors="coerce").fillna(0.0)
        df["is_int"] = df["Amount"].apply(lambda x: x.is_integer())
    if "Member Age" in df.columns:
        df["Member Age"] = pd.to_numeric(df["Member Age"], errors="coerce").fillna(0).astype(float)

    # Derive account_age_days if possible
    if "Account Open Date" in df.columns and "Post Date" in df.columns:
        df["Account Open Date"] = pd.to_datetime(df["Account Open Date"], errors="coerce")
        account_age = (df["Post Date"] - df["Account Open Date"]).dt.days
        df["account_age_days"] = account_age.fillna(0).astype(int)
    else:
        df["account_age_days"] = 0

    # Labels: use Fraud (ignore Fraud Description)
    if "Fraud" not in df.columns:
        # Default to zeros if not present
        df["Fraud"] = 0
    df["Fraud"] = pd.to_numeric(df["Fraud"], errors="coerce").fillna(0).astype(int)

    # Sort by available time keys (and id if存在)
    sort_cols = [c for c in ["Account ID", "Post Date", "Post Time"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)

    return df


def build_sequences_from_dataframe(df: pd.DataFrame):
    # Use ALL columns except label/desc/id as features
    exclude_cols = {"Fraud", "Fraud Description", "Account ID", "Fraud Adjustment Indicator"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 1) Post Date → day-of-year [0, 365]
    if "Post Date" in df.columns and "Post Date" in feature_cols:
        # ensure datetime
        post_date_dt = pd.to_datetime(df["Post Date"], errors="coerce")
        df["Post Date_doy"] = post_date_dt.dt.dayofyear.sub(1).clip(lower=0).fillna(0).astype("int64")
        feature_cols.append("Post Date_doy")
        feature_cols = [c for c in feature_cols if c != "Post Date"]

    # 2) Account Open Date → day-of-year [0, 365]
    if "Account Open Date" in df.columns and "Account Open Date" in feature_cols:
        aod_dt = pd.to_datetime(df["Account Open Date"], errors="coerce")
        df["Account Open Date_doy"] = aod_dt.dt.dayofyear.sub(1).clip(lower=0).fillna(0).astype("int64")
        feature_cols.append("Account Open Date_doy")
        feature_cols = [c for c in feature_cols if c != "Account Open Date"]

    # 3) Post Time → hour [0,24], minute [0,60] (ignore seconds)
    def _parse_hh_mm_from_time_text(x):
        try:
            if pd.isna(x):
                return np.nan, np.nan
            s = str(x)
            parts = s.split(":")
            if len(parts) >= 2:
                h = int(parts[0])
                m = int(parts[1])
                return h, m
        except Exception:
            pass
        return np.nan, np.nan

    post_time_hour = None
    post_time_min = None

    if "Post Time Parsed" in df.columns and "Post Time Parsed" in feature_cols:
        hh_mm = df["Post Time Parsed"].apply(lambda t: _parse_hh_mm_from_time_text(t))
        post_time_hour = hh_mm.apply(lambda x: x[0])
        post_time_min = hh_mm.apply(lambda x: x[1])
        feature_cols = [c for c in feature_cols if c != "Post Time Parsed"]
    elif "Post Time" in df.columns and "Post Time" in feature_cols:
        # Post Time numeric encoded as HHMMSS
        pt = pd.to_numeric(df["Post Time"], errors="coerce")
        post_time_hour = (pt // 10000).fillna(0).astype("int64")
        post_time_min = ((pt // 100) % 100).fillna(0).astype("int64")
        feature_cols = [c for c in feature_cols if c != "Post Time"]

    if post_time_hour is not None and post_time_min is not None:
        df["Post Time_hour"] = pd.to_numeric(post_time_hour, errors="coerce").fillna(0).clip(lower=0, upper=24).astype("int64")
        df["Post Time_minute"] = pd.to_numeric(post_time_min, errors="coerce").fillna(0).clip(lower=0, upper=60).astype("int64")
        feature_cols.extend(["Post Time_hour", "Post Time_minute"])

    # Remove any remaining datetime columns from features (we don't want raw datetimes)
    datetime_cols = []
    for col in list(feature_cols):
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
    feature_cols = [c for c in feature_cols if c not in datetime_cols]

    # To numeric for objects; factorize if still non-numeric
    processed = {}
    for col in feature_cols:
        series = df[col]
        if series.dtype == object:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                processed[col] = numeric
            else:
                codes, _ = pd.factorize(series.astype(str), sort=True)
                processed[col] = pd.Series(codes, index=series.index, dtype=np.int64)
        elif np.issubdtype(series.dtype, np.bool_):
            processed[col] = series.astype(np.int64)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            # Special handling for Amount: log-transform before scaling
            if col == "Amount":
                print("Amount log and scale")
                amt = numeric.fillna(0)
                processed[col] = np.sign(amt) * np.log1p(np.abs(amt))
            else:
                processed[col] = numeric

    X_df = pd.DataFrame(processed)
    X_df = X_df.fillna(0)

    # Scale all features
    scaler = StandardScaler().fit(X_df.values.astype(np.float32))
    X_scaled = scaler.transform(X_df.values.astype(np.float32))

    sequences = []
    if "Account ID" in df.columns:
        grouped = df.groupby("Account ID")
        for _, group in grouped:
            idx = group.index
            X = X_scaled[idx, :]
            y = group["Fraud"].values
            sequences.append((X, y))
    else:
        X = X_scaled
        y = df["Fraud"].values
        sequences.append((X, y))

    return sequences, list(X_df.columns)


class FraudDataset(Dataset):
    def __init__(self, sequences, max_len: int = 50):
        self.data = []
        for X, y in sequences:
            if len(X) > max_len:
                X, y = X[-max_len:], y[-max_len:]
            pad_len = max_len - len(X)
            # Pad at the beginning with zeros
            X_pad = np.vstack([np.zeros((pad_len, X.shape[1]), dtype=np.float32), X.astype(np.float32)])
            y_pad = np.hstack([np.zeros(pad_len, dtype=np.float32), y.astype(np.float32)])
            # Mask: 0 for padding, 1 for valid steps
            mask = np.hstack([np.zeros(pad_len, dtype=np.float32), np.ones(len(X), dtype=np.float32)])
            self.data.append((torch.tensor(X_pad, dtype=torch.float32),
                              torch.tensor(y_pad, dtype=torch.float32),
                              torch.tensor(mask, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Public API to build DataLoader from matched CSVs
def create_dataloader(matched_dir: str = "matched", max_len: int = 50, batch_size: int = 32, shuffle: bool = True, mode: str = "train", split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    df = load_matched_dataframe(matched_dir, mode=mode, split=split, seed=seed)
    sequences, feature_names = build_sequences_from_dataframe(df)
    dataset = FraudDataset(sequences, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, feature_names


if __name__ == "__main__":
    # Quick test example for creating datasets from matched CSVs
    try:
        for mode in ["train", "val", "test"]:
            print("---", mode, "---")
            loader, feature_names = create_dataloader(
                matched_dir="matched",
                max_len=50,
                batch_size=32,
                shuffle=(mode == "train"),
                mode=mode,
                split=(0.8, 0.1, 0.1),
                seed=42,
            )
            print("Features:", feature_names)
            for batch_idx, (X, y, mask) in enumerate(loader):
                print("Batch", batch_idx, "X shape:", tuple(X.shape), "y shape:", tuple(y.shape), "mask shape:", tuple(mask.shape))
                break
            # only inspect the first batch
            
    except Exception as e:
        print("Failed to create dataloader:", e)