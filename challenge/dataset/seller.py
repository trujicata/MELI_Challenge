import pandas as pd

from challenge.dataset.utils import typical_string_processing

POPULAR_USED_SELLERS = [
    6645536870,
    3846095764,
    6832522378,
    7394214410,
    4014611326,
    4755818264,
    8918336477,
    2024252139,
    4631246902,
    4226059250,
    4838664540,
    8220549814,
    8516415845,
    5255978862,
    8326621157,
    7740664679,
    6972484560,
    6537535599,
    6538753635,
    3135396608,
    6029888950,
    2122542660,
    6884045011,
    5749908984,
    7694743641,
    4248718919,
]

POPULAR_NEW_SELLERS = [
    8980863521,
    5248662274,
    3884593281,
    7106323686,
    2266082781,
    8612126795,
    2373910598,
    8435804226,
    7835318510,
    7060837357,
    7772844348,
    6396603751,
    7704929703,
    2015548469,
    6846806944,
    7394370231,
    1387735603,
]


def process_city_string(x: str) -> str:
    x = typical_string_processing(x)
    x = x.replace(".", "")
    x = x.replace("federal", "")
    if x == "ciudad autonoma de buenos aires":
        x = "caba"
    elif x == "bs as":
        x = "capital"
    elif x == "buenos aires":
        x = "capital"
    x = typical_string_processing(x)
    return x


def preprocess_seller_address(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the seller address column.
    It creates three new columns:
    - seller_address_state
    - seller_address_city
    - most_used_seller_address_cities
    - most_used_seller_address_states

    It drops the seller_address column.

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe

    """
    df["seller_address_state"] = df["seller_address"].apply(
        lambda x: x["state"]["name"]
    )
    df["seller_address_city"] = df["seller_address"].apply(lambda x: x["city"]["name"])
    df["seller_address_city"] = df["seller_address_city"].apply(process_city_string)
    df["most_used_seller_address_cities"] = df["seller_address_city"].apply(
        lambda x: x
        in [
            "bragado",
            "san andres de giles",
            "coghlan",
            "san antonio de padua",
            "la boca",
            "villa ortuzar",
        ]
    )
    df["most_new_seller_address_cities"] = df["seller_address_city"].apply(
        lambda x: x
        in [
            "munro",
            "villa crespo",
            "villa adelina",
            "trelew",
            "jose c paz",
            "mataderos",
            "villa santa rita",
            "ciudad madero",
        ]
    )
    df.drop(columns=["seller_address_city"], inplace=True)
    return df


def preprocess_seller_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the seller id column.

    It creates two new columns:
    - seller_id_most_used: if the seller is a popular used item seller
    - seller_id_most_new: if the seller is a popular new item seller

    It drops the seller_id column.

    Args:
        df: pd.DataFrame, the dataframe to preprocess
    """
    df["seller_id"] = df["seller_id"].apply(
        lambda x: (
            "popular used seller"
            if x in POPULAR_USED_SELLERS
            else "popular new seller" if x in POPULAR_NEW_SELLERS else "other"
        )
    )
    return df
