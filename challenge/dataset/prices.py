import numpy as np
import pandas as pd
from argendolar import Argendolar, TipoDivisas

argendolar = Argendolar()
dolar_oficial = argendolar.get_dolar_historia_completa(tipo=TipoDivisas.OFICIAL)


def transform_into_USD(price: float, currency_id: str, last_updated: str):
    if currency_id == "USD":
        return price
    else:
        # ensure last_updated is string YYYY-MM-DD
        if hasattr(last_updated, "strftime"):
            date_str = last_updated.strftime("%Y-%m-%d")
        else:
            date_str = str(last_updated)[:10]

        # get dolar value
        dolar_row = dolar_oficial.loc[dolar_oficial["fecha"] == date_str, "compra"]
        if dolar_row.empty:
            return None  # or np.nan
        dolar_price = dolar_row.iloc[0]
        return price / dolar_price


def standardize_prices(x: float) -> float:
    """
    This function standardizes the prices into USD.
    Using the mean and standard deviation of the log1p of the
    prices in USD from training data.

    Args:
        x: float, the price to standardize

    Returns:
        float, the standardized price
    """
    return (x - 7.952252) / 1.729920


def preprocess_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the prices column.
    It transforms the prices into USD using the argendolar library.

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """
    df["price_usd"] = df.apply(
        lambda row: transform_into_USD(
            row["price"], row["currency_id"], row["last_updated"]
        ),
        axis=1,
    )
    df["price_usd"] = df["price_usd"].apply(lambda x: np.log1p(x))
    df["price_usd"] = df["price_usd"].apply(lambda x: standardize_prices(x))

    df.drop(columns=["price", "currency_id", "last_updated"], inplace=True)
    return df
