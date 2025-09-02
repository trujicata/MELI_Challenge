import pandas as pd


def preprocess_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the categories column.
    It creates a new column with the popular categories with True or False.
    It drops the category_id column.

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """
    popular_categories = df["category_id"].value_counts().loc[lambda x: x > 100].index
    df["popular_category"] = df["category_id"].apply(lambda x: x in popular_categories)
    df.drop(columns=["category_id"], inplace=True)
    return df
