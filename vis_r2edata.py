data_path = "/minimax-dialogue/users/ruobai/rl_r2e/data/r2e_lite/dev.parquet"

import pandas as pd


def vis_r2edata(data_path):
    df = pd.read_parquet(data_path)
    print(df)
    print(df.columns)
    print(df.iloc[0]['extra_info']['ds'].keys())

if __name__ == "__main__":
    vis_r2edata(data_path)