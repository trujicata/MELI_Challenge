import pandas as pd


def dflt_bool(x: list) -> int:
    return len([m for m in x if m["attribute_group_id"] == "DFLT"]) > 0


def preprocess_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the attributes column.

    Finds if there's attributes with the DFLT group id

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """
    df["attributes_dflt"] = df["attributes"].apply(dflt_bool)
    df.drop(columns=["attributes"], inplace=True)
    return df
