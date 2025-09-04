import pandas as pd
from datetime import datetime


def preprocess_times(df: pd.DataFrame) -> pd.DataFrame:

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["stop_time"] = pd.to_datetime(df["stop_time"])

    df["total_time"] = df["stop_time"] - df["start_time"]
    df["total_time_seconds"] = df["total_time"].dt.total_seconds()

    df.drop(columns=["start_time", "stop_time", "total_time"], inplace=True)
    return df
