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
    批量读取多个 CSV 并拼接为一个 DataFrame。

    Args:
        files: 明确的文件名列表（相对 path）。如 ["a.csv", "b.csv"]
        glob_pattern: 通配符（相对 path）。如 "matched/*.csv"
        path: 基础目录
        sample: 每个文件读取的行数；-1 表示读全量
        add_source: 是否添加来源文件列 `__source_file__`
        join: 拼接列策略，"outer" 或 "inner"
        axis: 拼接轴，默认 0（纵向追加）
        read_kwargs: 透传给 pandas.read_csv 的额外参数(dict)

    Returns:
        pd.DataFrame
    """
    if files is None and glob_pattern is None:
        raise ValueError("必须提供 files 或 glob_pattern 之一")

    read_kwargs = read_kwargs or {}
    # 确保稳定列推断
    if "low_memory" not in read_kwargs:
        read_kwargs["low_memory"] = False

    # 解析文件列表
    file_paths = []
    if files:
        file_paths.extend([os.path.join(path, f) for f in files])
    if glob_pattern:
        file_paths.extend(glob.glob(os.path.join(path, glob_pattern)))

    if not file_paths:
        raise FileNotFoundError("未找到任何可读取的文件")

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
