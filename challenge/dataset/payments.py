import pandas as pd

TARJETAS = [
    "Tarjeta de crédito",
    "Visa Electron",
    "Mastercard Maestro",
    "American Express",
    "Mastercard",
    "Diners",
    "Visa",
]


def preprocess_non_mercadopago_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the non mercado pago payment methods column.
    It creates the following columns:
    - tarjeta
    - efectivo
    - mercado_pago
    - transferencia_bancaria
    - acordar_comprador
    - num_payment_methods_non_mercadopago

    Args:
        df: pd.DataFrame, the dataframe to preprocess

    Returns:
        pd.DataFrame, the preprocessed dataframe
    """

    df["tarjeta"] = df["non_mercado_pago_payment_methods"].apply(
        lambda x: any(m["description"] in TARJETAS for m in x)
    )

    df["efectivo"] = df["non_mercado_pago_payment_methods"].apply(
        lambda x: any(m["description"] == "Efectivo" for m in x)
    )
    df["transferencia_bancaria"] = df["non_mercado_pago_payment_methods"].apply(
        lambda x: any(m["description"] == "Transferencia bancaria" for m in x)
    )
    df["acordar_comprador"] = df["non_mercado_pago_payment_methods"].apply(
        lambda x: any(m["description"] == "Acordar con el comprador" for m in x)
    )
    df["num_payment_methods_non_mercadopago"] = df[
        "non_mercado_pago_payment_methods"
    ].apply(lambda x: len(x))
    df.drop(columns=["non_mercado_pago_payment_methods"], inplace=True)

    return df
