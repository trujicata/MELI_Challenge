import numpy as np
import pandas as pd


def preprocess_quantities(
    df: pd.DataFrame, quantity_column: str, possible_zeros: bool = False
) -> pd.DataFrame:
    """
    This function is used to preprocess the quantities of the items.
    Uses logarithm of the quantities to scale the data.

    Args:
        df: pd.DataFrame, the dataframe to preprocess
        quantity_column: str, the column to preprocess
        possible_zeros: bool, if True, some of the quantities are 0.

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """
    if possible_zeros:
        df[quantity_column] = np.log1p(df[quantity_column])
    else:
        df[quantity_column] = np.log(df[quantity_column])
    return df
