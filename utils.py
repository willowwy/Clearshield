import pandas as pd
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


def filter_history(df, account, mode, sample = 20):
    # account id, mode (0 for all info, 1 for all transactions in,  2 for all transactions out), top k entries
    if mode == 0:
        return df[df['Account ID']==account].sort_values(by="col_name")
if __name__ == "__main__":
    df = load_data("datasample.csv", sample = -1)
    # print(df.iloc[0]['Account ID'])
    print(filter_history(df, df.iloc[12]['Account ID'], 0))
