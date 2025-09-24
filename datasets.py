import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import mstats


def _discretize_amount(amount_series: pd.Series, n_bins: int = 12, percentile: float = 99.5) -> pd.Series:
    """
    金额离散化策略：截尾 → 对数变换 → 等宽分箱
    
    Args:
        amount_series: 原始金额数据
        n_bins: 分箱数量
        percentile: 截尾百分位数
    
    Returns:
        离散化后的金额编码 (0 到 n_bins-1)
    """
    # 1. 截尾处理：按指定百分位数 winsorize
    p995 = np.percentile(amount_series, percentile)
    amount_winsorized = np.minimum(amount_series, p995)
    
    # 2. 对数变换：log1p 处理，避免 log(0) 问题
    amount_log = np.log1p(amount_winsorized)
    
    # 3. 等宽分箱：在 log 空间上做等宽切分
    # 找到 log 空间的最小值和最大值
    log_min = amount_log.min()
    log_max = amount_log.max()
    
    # 创建等宽分箱边界
    bin_edges = np.linspace(log_min, log_max, n_bins + 1)
    
    # 分箱：使用 pd.cut 进行离散化
    amount_binned = pd.cut(amount_log, bins=bin_edges, labels=False, include_lowest=True)
    
    # 处理 NaN 值（边界情况）
    amount_binned = amount_binned.fillna(0).astype(int)
    
    return amount_binned


def _quantize_account_age(account_age_days: pd.Series) -> pd.Series:
    """
    账户年龄量子化：将账户年龄（天数）分为几个阶段
    
    Args:
        account_age_days: 账户年龄（天数）
    
    Returns:
        量子化后的年龄编码 (0-4)
        0: 1年以下 (< 365天)
        1: 1-2年 (365-729天)
        2: 3-5年 (730-1824天)
        3: 5-10年 (1825-3650天)
        4: 10年以上 (> 3650天)
    """
    age_quantized = pd.Series(0, index=account_age_days.index, dtype=int)
    
    # 1年以下
    age_quantized[account_age_days < 365] = 0
    # 1-2年
    age_quantized[(account_age_days >= 365) & (account_age_days < 730)] = 1
    # 3-5年
    age_quantized[(account_age_days >= 730) & (account_age_days < 1825)] = 2
    # 5-10年
    age_quantized[(account_age_days >= 1825) & (account_age_days < 3651)] = 3
    # 10年以上
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
    处理单个DataFrame的数据清洗和特征工程
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
        # 添加量子化的账户年龄
        df["account_age_quantized"] = _quantize_account_age(df["account_age_days"])
    else:
        df["account_age_days"] = 0
        df["account_age_quantized"] = 0

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


def load_matched_dataframes(matched_dir: str = "matched", mode: str = "train", split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> list[pd.DataFrame]:
    """
    加载匹配的CSV文件，返回DataFrame列表而不是拼接的DataFrame
    确保每个DataFrame对应一个CSV文件，保持数据的独立性
    """
    chosen_files = _select_csv_files(matched_dir, mode, split, seed)
    print(f"Loading {len(chosen_files)} files for {mode} mode")
    
    dataframes = []
    for path in chosen_files:
        df = pd.read_csv(path)
        df = _process_single_dataframe(df)
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
                # 跳过account_age_days特征，不使用它
                continue
            else:
                processed[col] = numeric

    X_df = pd.DataFrame(processed)
    X_df = X_df.fillna(0)

    # 不进行归一化，直接使用原始值（后续会使用embedding层）
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
    def __init__(self, sequences, max_len: int = 50):
        """
        sequences: list of (X, y) tuples where each tuple represents sequences from a single CSV file
        """
        self.data = []
        self.csv_file_indices = []  # 记录每个序列来自哪个CSV文件
        
        for csv_idx, (X, y) in enumerate(sequences):
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
            self.csv_file_indices.append(csv_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_csv_file_indices(self):
        """返回每个序列对应的CSV文件索引"""
        return self.csv_file_indices


class CSVConsistentBatchSampler:
    """
    自定义采样器，确保每个batch只包含来自同一个CSV文件的数据
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.csv_file_indices = dataset.get_csv_file_indices()
        
        # 按CSV文件分组
        self.csv_groups = {}
        for idx, csv_idx in enumerate(self.csv_file_indices):
            if csv_idx not in self.csv_groups:
                self.csv_groups[csv_idx] = []
            self.csv_groups[csv_idx].append(idx)
    
    def __iter__(self):
        # 为每个CSV文件创建batch
        for csv_idx, indices in self.csv_groups.items():
            if self.shuffle:
                np.random.shuffle(indices)
            
            # 将同一CSV文件的序列分批
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield batch_indices
    
    def __len__(self):
        total_batches = 0
        for indices in self.csv_groups.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


# Public API to build DataLoader from matched CSVs
def create_dataloader(matched_dir: str = "matched", max_len: int = 50, batch_size: int = 32, shuffle: bool = True, mode: str = "train", split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    """
    创建数据加载器，确保每个batch只包含来自同一个CSV文件的数据
    """
    dataframes = load_matched_dataframes(matched_dir, mode=mode, split=split, seed=seed)
    
    all_sequences = []
    feature_names = None
    
    # 为每个CSV文件构建序列
    for df in dataframes:
        sequences, feature_names = build_sequences_from_dataframe(df)
        all_sequences.extend(sequences)
    
    dataset = FraudDataset(all_sequences, max_len=max_len)
    
    # 使用自定义采样器确保batch一致性
    batch_sampler = CSVConsistentBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    
    return loader, feature_names



if __name__ == "__main__":
    # Quick test example for creating datasets from matched CSVs
    try:
        # 测试多文件模式
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
            print("Total batches:", len(loader))
            
            # 检查前几个batch，验证batch一致性
            for batch_idx, (X, y, mask) in enumerate(loader):
                print("Batch", batch_idx, "X shape:", tuple(X.shape), "y shape:", tuple(y.shape), "mask shape:", tuple(mask.shape))
                if batch_idx >= 2:  # 只检查前3个batch
                    break
            
    except Exception as e:
        print("Failed to create dataloader:", e)
        import traceback
        traceback.print_exc()