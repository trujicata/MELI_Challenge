import pandas as pd
import numpy as np


def calculate_total_max_size(x: str):
    max_sizes = list(set([m["max_size"] for m in x]))
    max_size = 0
    for size in max_sizes:
        try:
            height = int(size.split("x")[0])
            width = int(size.split("x")[1])
            if height * width > max_size:
                max_size = height * width
        except:
            pass
    return max_size


def preprocess_pictures(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the pictures column.
    It creates a new column with the total max size of the pictures.
    It drops the pictures column.

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """

    df["pictures_max_size"] = df["pictures"].apply(calculate_total_max_size)
    df["pictures_max_size"] = df["pictures_max_size"].apply(lambda x: np.log1p(x))
    df.drop(columns=["pictures"], inplace=True)
    return df
