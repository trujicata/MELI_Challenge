import pandas as pd


def check_methods(value: dict):
    """
    This function checks if the shipping methods are present in the dictionary.
    It returns the methods if they are present, otherwise it returns None.

    Args:
        value: dict, the dictionary to check

    Returns:
        list, the methods if they are present, otherwise it returns None.
    """
    if "methods" in value.keys():
        return value["methods"]
    elif "free_methods" in value.keys():
        return value["free_methods"]
    else:
        return None


def preprocess_shipping(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the shipping column.
    It creates four new columns:
    - shipping_local_pick_up
    - shipping_free_shipping
    - shipping_method
    - shipping_mode

    It drops the shipping column.

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """

    df["shipping_local_pick_up"] = df["shipping"].apply(lambda x: x["local_pick_up"])
    df["shipping_free_shipping"] = df["shipping"].apply(lambda x: x["free_shipping"])
    df["shipping_method"] = (
        df["shipping"]
        .apply(check_methods)
        .apply(
            lambda x: "free_mode" if isinstance(x, list) and len(x) > 0 else "unknown"
        )
    )
    df["shipping_mode"] = df["shipping"].apply(lambda x: x["mode"])
    df.drop(columns=["shipping"], inplace=True)
    return df
