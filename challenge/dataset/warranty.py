from typing import Literal
import pandas as pd

from challenge.dataset.utils import typical_string_processing


def find_warranty_magnitude(x: str) -> str:
    """
    This function finds the magnitude of the warranty.
    It checks if the magnitude is in years, months, weeks or days.

    Args:
        x: str, the string to process

    Returns:
        str, the processed string


    """
    if "año" in x or "ano" in x.split(" ") or "anos" in x.split(" "):
        return "años"
    elif "mes" in x:
        return "meses"
    elif "semana" in x:
        return "semanas"
    elif "dia" in x:
        try:
            days = int(x.split("dia")[0].split(" ")[-2])
            if days > 365:
                return "años"
            elif days > 30:
                return "meses"
            elif days > 7:
                return "semanas"
            else:
                return "dias"
        except (ValueError, IndexError):
            return x
    elif "horas" in x:
        return "dias"  # Hypothesis: the warranty is for 24 or 48 hours
    else:
        return x


def warranty_de_fabricacion(x: str) -> str:
    """
    This function checks if the warranty is from the factory.

    Args:
        x: str, the string to process

    Returns:
        str, the processed string
    """
    if "fabrica" in x or "origen" in x:  # Hypothesis: the warranty is from the factory
        return "de fabrica"
    else:
        return x


def yes_no_warranty(x):
    if x not in ["años", "meses", "semanas", "dias", "de fabrica", "usado", "a nuevo"]:
        x = x.replace(".", "")
        if "si" in x.split(" ") or "con garantia" in x or "garantia total" in x:
            return "si"
        elif "sin garantia" in x:
            return "no"
        else:
            return "otro"
    else:
        return x


def warranty_estado(x: str) -> str:
    """
    This function checks if in the warranty the seller mentions the state of the item.

    Args:
        x: str, the string to process

    Returns:
        str, the processed string. Can be "a nuevo", "usado", "otro", "de fabrica", "años", "meses", "semanas", "dias", "si", "no"
    """
    if x not in ["años", "meses", "semanas", "dias", "de fabrica"]:
        x = typical_string_processing(x)
        if "nuevo" in x:
            return "a nuevo"
        elif "usado" in x:
            if "devolucion" not in x or "sido" not in x:
                return "usado"
            else:
                return "otro"
        elif "reparado" in x:
            return "usado"
        elif (
            "estado" in x or "excelentes condiciones" in x or "buenas condiciones" in x
        ):
            return "usado"
        else:
            return x
    else:
        return x


def warranty_string_processing(
    x: str,
) -> Literal[
    "a nuevo",
    "usado",
    "otro",
    "de fabrica",
    "años",
    "meses",
    "semanas",
    "dias",
    "si",
    "no",
    "unknown",
]:
    """
    This function processes the warranty string.
    It removes accent marks, converts to lowercase, removes dots and trims the string.
    It finds the magnitude of the warranty.
    It checks if the warranty is from the factory.
    It checks if in the warranty the seller mentions the state of the item.

    Args:
        x: str, the string to process

    Returns:
        str, the processed string.
        Can be "a nuevo", "usado", "otro", "de fabrica", "años", "meses", "semanas", "dias", "si", "no","unknown"
    """
    if isinstance(x, str) and x != "unknown":
        x = typical_string_processing(x)
        x = find_warranty_magnitude(x)
        x = warranty_de_fabricacion(x)
        x = warranty_estado(x)
        x = yes_no_warranty(x)
    return x


def preprocess_warranty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the warranty column.

    It modifiy the warranty string, doing:
    - Remove accent marks, convert to lowercase, remove dots and trim the string
    - Find the magnitude of the warranty
    - Check if the warranty is from the factory
    - Check if in the warranty the seller mentions the state of the item
    - Check if the warranty is a new warranty or a used warranty
    - Check if the warranty is si or no
    - Check if the warranty is other

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """

    df["warranty"] = df["warranty"].fillna("unknown")
    df["warranty"] = df["warranty"].apply(lambda x: "unknown" if x == "" else x)
    df["warranty"] = df["warranty"].apply(warranty_string_processing)
    return df
