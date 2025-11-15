import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import mstats


def _discretize_amount(amount_series: pd.Series, n_bins: int = 12, percentile: float = 99.5) -> pd.Series:
    """
    Amount discretization strategy: winsorization → log transformation → equal-width binning
    
    Args:
        amount_series: Original amount data
        n_bins: Number of bins
        percentile: Winsorization percentile
    
    Returns:
        Discretized amount encoding (0 to n_bins-1)
    """
    # Guard: empty series
    if amount_series is None or len(amount_series) == 0:
        return pd.Series([], index=(amount_series.index if isinstance(amount_series, pd.Series) else None), dtype=int)

    # 1. Winsorization: clip at specified percentile
    p995 = np.percentile(amount_series, percentile)
    amount_winsorized = np.minimum(amount_series, p995)
    
    # 2. Log transformation: log1p to avoid log(0) issues
    amount_log = np.log1p(amount_winsorized)
    
    # 3. Equal-width binning: split in log space
    # Find min and max values in log space
    log_min = amount_log.min()
    log_max = amount_log.max()
    
    # Handle special case: if all values are the same, return 0
    if log_min == log_max:
        return pd.Series(0, index=amount_series.index, dtype=int)
    
    # Create equal-width bin edges
    bin_edges = np.linspace(log_min, log_max, n_bins + 1)
    
    # Binning: use pd.cut for discretization, handle duplicate boundaries
    amount_binned = pd.cut(amount_log, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
    
    # Handle NaN values (boundary cases)
    amount_binned = amount_binned.fillna(0).astype(int)
    
    return amount_binned


def _quantize_account_age(account_age_days: pd.Series) -> pd.Series:
    """
    Account age quantization: divide account age (days) into several stages
    
    Args:
        account_age_days: Account age in days
    
    Returns:
        Quantized age encoding (0-4)
        0: Less than 1 year (< 365 days)
        1: 1-2 years (365-729 days)
        2: 3-5 years (730-1824 days)
        3: 5-10 years (1825-3650 days)
        4: More than 10 years (> 3650 days)
    """
    age_quantized = pd.Series(0, index=account_age_days.index, dtype=int)
    
    # Less than 1 year
    age_quantized[account_age_days < 365] = 0
    # 1-2 years
    age_quantized[(account_age_days >= 365) & (account_age_days < 730)] = 1
    # 3-5 years
    age_quantized[(account_age_days >= 730) & (account_age_days < 1825)] = 2
    # 5-10 years
    age_quantized[(account_age_days >= 1825) & (account_age_days < 3651)] = 3
    # More than 10 years
    age_quantized[account_age_days >= 3651] = 4
    
    return age_quantized


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

def _process_single_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process single DataFrame for data cleaning and feature engineering
    """
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
        # Add quantized account age
        df["account_age_quantized"] = _quantize_account_age(df["account_age_days"])
    else:
        df["account_age_days"] = 0
        df["account_age_quantized"] = 0

    # Labels: use Fraud (ignore Fraud Description)
    if "Fraud" not in df.columns:
        # Default to zeros if not present
        df["Fraud"] = 0
    df["Fraud"] = pd.to_numeric(df["Fraud"], errors="coerce").fillna(0).astype(int)

    # Sort by available time keys (and id if exists)
    sort_cols = [c for c in ["Account ID", "Post Date", "Post Time"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)

    return df


def load_matched_dataframes(matched_dir: str = "matched", mode: str = "train", split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> list[pd.DataFrame]:
    """
    Load matched CSV files, return DataFrame list instead of concatenated DataFrame
    Ensure each DataFrame corresponds to one CSV file, maintaining data independence
    """
    chosen_files = _select_csv_files(matched_dir, mode, split, seed)
    print(f"Loading {len(chosen_files)} files for {mode} mode")
    
    dataframes = []
    for path in chosen_files:
        df = pd.read_csv(path)
        # Skip empty DataFrame (no rows)
        if df.shape[0] == 0:
            continue
        df = _process_single_dataframe(df)
        # Skip after processing if still empty or all-NaN rows
        if df.shape[0] == 0:
            continue
        dataframes.append(df)
    
    return dataframes


def build_sequences_from_dataframe(df: pd.DataFrame):
    # Use ALL columns except label/desc/id as features
    exclude_cols = {"Fraud", "Fraud Description", "Account ID", "Fraud Adjustment Indicator"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 1) Post Date → day-of-year [0, 365]
    if "Post Date" in df.columns and "Post Date" in feature_cols:
        # ensure datetime
        post_date_dt = pd.to_datetime(df["Post Date"], errors="coerce")
        df["Post Date_doy"] = post_date_dt.dt.dayofyear.sub(1).clip(lower=0, upper=365).fillna(0).astype("int64")
        feature_cols.append("Post Date_doy")
        feature_cols = [c for c in feature_cols if c != "Post Date"]

    # 2) Account Open Date → day-of-year [0, 365]
    if "Account Open Date" in df.columns and "Account Open Date" in feature_cols:
        aod_dt = pd.to_datetime(df["Account Open Date"], errors="coerce")
        df["Account Open Date_doy"] = aod_dt.dt.dayofyear.sub(1).clip(lower=0, upper=365).fillna(0).astype("int64")
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
        df["Post Time_hour"] = pd.to_numeric(post_time_hour, errors="coerce").fillna(0).clip(lower=0, upper=23).astype("int64")
        df["Post Time_minute"] = pd.to_numeric(post_time_min, errors="coerce").fillna(0).clip(lower=0, upper=59).astype("int64")
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
            # Special handling for Amount: discretization strategy
            if col == "Amount":
                amt = numeric.fillna(0)
                processed[col] = _discretize_amount(amt )
            elif col == "account_age_days":
                # Skip account_age_days feature, don't use it
                continue
            else:
                processed[col] = numeric

    X_df = pd.DataFrame(processed)
    X_df = X_df.fillna(0)

    # No normalization, use raw values directly (embedding layers will be used later)
    X_scaled = X_df.values.astype(np.float32)

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
    def __init__(self, sequences_with_csv_idx, max_len: int = 50, use_sliding_window: bool = False, window_overlap: float = 0.5):
        """
        sequences_with_csv_idx: list of (csv_idx, X, y) tuples where csv_idx indicates which CSV file the sequence comes from
        max_len: Maximum sequence length
        use_sliding_window: Whether to use sliding window
        window_overlap: Sliding window overlap ratio (0.0-0.9)
        """
        self.data = []
        self.csv_file_indices = []  # Record which CSV file each sequence comes from
        self.max_len = max_len
        self.use_sliding_window = use_sliding_window
        self.window_overlap = window_overlap
        
        for csv_idx, X, y in sequences_with_csv_idx:
            seq_len = len(X)
            if seq_len <= 0:
                continue

            if self.use_sliding_window and seq_len > max_len:
                # Sliding window strategy: overlapping sampling, significantly increase data volume
                step_size = max(1, int(max_len * (1 - window_overlap)))
                for start in range(0, seq_len - max_len + 1, step_size):
                    end = start + max_len
                    self._add_sequence(X[start:end], y[start:end], csv_idx)
            else:
                # Original strategy: non-overlapping segmentation
                for start in range(0, seq_len, max_len):
                    end = min(start + max_len, seq_len)
                    X_seg = X[start:end]
                    y_seg = y[start:end]
                    self._add_sequence(X_seg, y_seg, csv_idx)

    def _add_sequence(self, X_seg, y_seg, csv_idx):
        """Add sequence to dataset"""
        pad_len = self.max_len - len(X_seg)
        
        # Left zero padding to max_len
        X_pad = np.vstack([
            np.zeros((pad_len, X_seg.shape[1]), dtype=np.float32),
            X_seg.astype(np.float32)
        ])
        y_pad = np.hstack([
            np.zeros(pad_len, dtype=np.float32),
            y_seg.astype(np.float32)
        ])
        # Mask: 0 for padding, 1 for valid steps
        mask = np.hstack([
            np.zeros(pad_len, dtype=np.float32),
            np.ones(len(X_seg), dtype=np.float32)
        ])

        self.data.append((
            torch.tensor(X_pad, dtype=torch.float32),
            torch.tensor(y_pad, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        ))
        self.csv_file_indices.append(csv_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_csv_file_indices(self):
        """Return CSV file index for each sequence"""
        return self.csv_file_indices
    
    def get_csv_file_distribution(self):
        """Return distribution of sequences across CSV files"""
        distribution = {}
        for csv_idx in self.csv_file_indices:
            distribution[csv_idx] = distribution.get(csv_idx, 0) + 1
        return distribution


class CSVConsistentBatchSampler:
    """
    Custom sampler to ensure each batch only contains data from the same CSV file
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_all_zero_batches: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_all_zero_batches = drop_all_zero_batches
        self.csv_file_indices = dataset.get_csv_file_indices()
        
        # Group by CSV file
        self.csv_groups = {}
        for idx, csv_idx in enumerate(self.csv_file_indices):
            if csv_idx not in self.csv_groups:
                self.csv_groups[csv_idx] = []
            self.csv_groups[csv_idx].append(idx)
    
    def __iter__(self):
        # Create batches for each CSV file
        for csv_idx, indices in self.csv_groups.items():
            if self.shuffle:
                np.random.shuffle(indices)
            
            # Batch sequences from the same CSV file
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if self.drop_all_zero_batches:
                    # Filter out batches where y is all zeros (consider mask, only count valid steps)
                    has_positive = False
                    for di in batch_indices:
                        _, y_t, mask_t = self.dataset[di]
                        if ((y_t == 1) & (mask_t == 1)).any().item():
                            has_positive = True
                            break
                    if not has_positive:
                        continue
                yield batch_indices
    
    def __len__(self):
        """
        Calculate total number of batches.
        Note: This is an upper bound if drop_all_zero_batches=True,
        as some batches may be filtered out during iteration.
        """
        total_batches = 0
        for indices in self.csv_groups.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


# Public API to build DataLoader from matched CSVs
def create_dataloader(matched_dir: str = "clustered_out/no_fraud", max_len: int = 50, batch_size: int = 32, shuffle: bool = True, mode: str = "train", split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42, drop_all_zero_batches: bool = True, use_sliding_window: bool = False, window_overlap: float = 0.5):
    """
    Create data loader with sliding window support to enhance data utilization efficiency
    
    Args:
        matched_dir: Data directory
        max_len: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        mode: Data mode (train/val/test)
        split: Data split ratios
        seed: Random seed
        drop_all_zero_batches: Whether to drop all-zero batches
        use_sliding_window: Whether to use sliding window
        window_overlap: Sliding window overlap ratio (0.0-0.9)
    """
    dataframes = load_matched_dataframes(matched_dir, mode=mode, split=split, seed=seed)
    
    all_sequences_with_csv_idx = []
    feature_names = None
    
    # Build sequences for each CSV file, assign unique csv_idx to each CSV file
    for csv_idx, df in enumerate(dataframes):
        sequences, feature_names = build_sequences_from_dataframe(df)
        # All sequences from the same CSV file share the same csv_idx
        for X, y in sequences:
            all_sequences_with_csv_idx.append((csv_idx, X, y))
    
    # Create dataset with sliding window support
    dataset = FraudDataset(
        all_sequences_with_csv_idx, 
        max_len=max_len,
        use_sliding_window=use_sliding_window and mode == "train",  # Only use sliding window during training
        window_overlap=window_overlap
    )
    
    # Use custom sampler to ensure batch consistency and optionally drop all-zero label batches
    batch_sampler = CSVConsistentBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_all_zero_batches=drop_all_zero_batches)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    
    return loader, feature_names



if __name__ == "__main__":
    # Test CSV file consistency in batches
    print("=" * 80)
    print("Test CSV File Consistency in Batches")
    print("=" * 80)
    
    try:
        loader, feature_names = create_dataloader(
            matched_dir="clustered_out/no_fraud",
            max_len=50,
            batch_size=32,
            shuffle=True,
            mode="train",
            split=(0.8, 0.1, 0.1),
            seed=42,
            use_sliding_window=False,
            window_overlap=0.0,
            drop_all_zero_batches=False  # Disable to see all batches
        )
        
        print(f"Features: {len(feature_names)} features")
        print(f"Total batches (from __len__): {len(loader)}")
        
        # Get dataset to access csv_file_indices
        dataset = loader.dataset
        csv_file_indices = dataset.get_csv_file_indices()
        
        print(f"Total sequences in dataset: {len(dataset)}")
        print(f"Unique CSV file indices: {len(set(csv_file_indices))}")
        
        # Check batch consistency
        batch_csv_groups = {}
        actual_batch_count = 0
        
        for batch_idx, (X, y, mask) in enumerate(loader):
            actual_batch_count += 1
            batch_size = X.shape[0]
            
            # Get the dataset indices for this batch
            # Note: We need to track which indices are in each batch
            # Since we're using a batch_sampler, we can't directly get the indices
            # But we can check if all sequences in a batch have the same csv_idx
            
            # For now, let's just print the first few batches
            if batch_idx < 10:
                print(f"\nBatch {batch_idx}: size={batch_size}")
            
        print(f"\nActual batches yielded: {actual_batch_count}")
        print(f"Expected batches (from __len__): {len(loader)}")
        
        # Verify CSV grouping in sampler
        batch_sampler = loader.batch_sampler
        print(f"\nCSV groups in sampler: {len(batch_sampler.csv_groups)}")
        print(f"Total sequences grouped: {sum(len(indices) for indices in batch_sampler.csv_groups.values())}")
        
        # Check CSV file distribution
        csv_file_distribution = {}
        for csv_idx in csv_file_indices:
            csv_file_distribution[csv_idx] = csv_file_distribution.get(csv_idx, 0) + 1
        
        print(f"\nCSV file distribution (how many sequences per CSV file):")
        for csv_idx, count in sorted(csv_file_distribution.items())[:10]:
            print(f"  CSV file {csv_idx}: {count} sequences")
        
        # Verify that sampler groups match the distribution
        print(f"\nVerifying sampler groups:")
        for csv_idx, indices in list(batch_sampler.csv_groups.items())[:5]:
            print(f"  Group csv_idx={csv_idx}: {len(indices)} sequences")
            # Check that all indices in this group have the same csv_idx
            csv_indices_in_group = [csv_file_indices[i] for i in indices]
            unique_csv_indices = set(csv_indices_in_group)
            if len(unique_csv_indices) == 1:
                print(f"    ✓ All sequences have csv_idx={list(unique_csv_indices)[0]}")
            else:
                print(f"    ✗ ERROR: Mixed csv_idx in group: {unique_csv_indices}")
                print(f"      This should not happen! Group contains sequences from different CSV files.")
            
    except Exception as e:
        print("Failed to create dataloader:", e)
        import traceback
        traceback.print_exc()