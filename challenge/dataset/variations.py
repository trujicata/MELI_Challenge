import numpy as np
import pandas as pd


def get_amt_var_attributes(x: list) -> int:
    """
    This function gets the amount attributesof variations.

    Args:
        x: list, the variations

    Returns:
        int, the amount of variations
    """
    if len(x) > 0:
        att_combs = 0
        for variation in x:
            att_combs += len(variation["attribute_combinations"])
        return att_combs
    else:
        return 0


def preprocess_variations(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the variations column.
    It transforms the variations column into the number of variations.

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """
    df["variations_amt_var_attributes"] = df["variations"].apply(get_amt_var_attributes)
    df["variations"] = df["variations"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    variations_amount_bins = [-1, 0, 1, 2, 5, 10, 200]
    df["variations_amount_bin"] = pd.cut(
        df["variations_amt_var_attributes"], bins=variations_amount_bins
    )
    df["variations_amount_bin"] = df["variations_amount_bin"].astype("category")

    variations_bins = [-1, 0, 2, 100]
    df["variations_bin"] = pd.cut(df["variations"], bins=variations_bins)
    df["variations_bin"] = df["variations_bin"].astype("category")

    df.drop(columns=["variations_amt_var_attributes", "variations"], inplace=True)
    return df
