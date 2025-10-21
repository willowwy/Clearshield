import pandas as pd
import glob
import os

def load_data(file, path = "./", sample = 20):
    # file name, file path, top n entries (-1 for all)
    try:
        if sample>0:
            df = pd.read_csv(path+file, nrows=sample)
        else:
            df = pd.read_csv(path+file)
    except Exception as e:
        raise Exception

    return df


def load_multiple(files=None, glob_pattern=None, path="./", sample=-1, add_source=False, join="outer", axis=0, read_kwargs=None):
    """
    Batch read multiple CSV files and concatenate into a single DataFrame.

    Args:
        files: Explicit list of file names (relative to path). e.g., ["a.csv", "b.csv"]
        glob_pattern: Glob pattern (relative to path). e.g., "matched/*.csv"
        path: Base directory
        sample: Number of rows to read from each file; -1 means read all
        add_source: Whether to add source file column `__source_file__`
        join: Concatenation strategy, "outer" or "inner"
        axis: Concatenation axis, default 0 (vertical append)
        read_kwargs: Additional parameters passed to pandas.read_csv (dict)

    Returns:
        pd.DataFrame
    """
    if files is None and glob_pattern is None:
        raise ValueError("Must provide either files or glob_pattern")

    read_kwargs = read_kwargs or {}
    # Ensure stable column inference
    if "low_memory" not in read_kwargs:
        read_kwargs["low_memory"] = False

    # Parse file list
    file_paths = []
    if files:
        file_paths.extend([os.path.join(path, f) for f in files])
    if glob_pattern:
        file_paths.extend(glob.glob(os.path.join(path, glob_pattern)))

    if not file_paths:
        raise FileNotFoundError("No readable files found")

    frames = []
    for fp in sorted(file_paths):
        kwargs = dict(read_kwargs)
        if sample > 0:
            kwargs["nrows"] = sample
        df = pd.read_csv(fp, **kwargs)
        if add_source:
            df["__source_file__"] = os.path.basename(fp)
        frames.append(df)

    return pd.concat(frames, axis=axis, join=join, ignore_index=True)


def filter_history(df, account, mode, sample = 20):
    # account id, mode (0 for all info, 1 for all transactions in,  2 for all transactions out), top k entries
    if mode == 0:
        return df[df['Account ID']==account].sort_values(by="col_name")
if __name__ == "__main__":
    df = load_data("datasample.csv", sample = -1)
    # print(df.iloc[0]['Account ID'])
    print(filter_history(df, df.iloc[12]['Account ID'], 0))
