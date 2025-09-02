# %%
import start  # noqa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from challenge.new_or_used import build_dataset

# %%
df = build_dataset()
df.head()

# %%
column = "non_mercado_pago_payment_methods"

# %%
df[column].describe()

# %%
df[column].isna().sum() + df[column].isnull().sum()

# %%
df[column].iloc[0]

# %%
df[column].value_counts()
# %%
# Check all the possible values of non_mercado_pago_payment_methods
all_methods = [
    m["description"]
    for methods in df["non_mercado_pago_payment_methods"]
    if methods
    for m in methods
]

# %%
all_methods = list(set(all_methods))
# %%
tarjetas = [
    "Tarjeta de crédito",
    "Visa Electron",
    "Mastercard Maestro",
    "American Express",
    "Mastercard",
    "Diners",
    "Visa",
]
df["tarjeta"] = df[column].apply(lambda x: any(m["description"] in tarjetas for m in x))
df["tarjeta"].value_counts()
# %%
df["efectivo"] = df[column].apply(
    lambda x: any(m["description"] == "Efectivo" for m in x)
)
df["efectivo"].value_counts()
# %%
df["mercadopago"] = df[column].apply(
    lambda x: any(m["description"] == "MercadoPago" for m in x)
)
df["mercadopago"].value_counts()

# %%
df["contra_reembolso"] = df[column].apply(
    lambda x: any(m["description"] == "Contra reembolso" for m in x)
)
df["contra_reembolso"].value_counts()

# %%
df["cheque_certificado"] = df[column].apply(
    lambda x: any(m["description"] == "Cheque certificado" for m in x)
)
df["cheque_certificado"].value_counts()

# %%
df["giro_postal"] = df[column].apply(
    lambda x: any(m["description"] == "Giro postal" for m in x)
)
df["giro_postal"].value_counts()

# %%
df["transferencia_bancaria"] = df[column].apply(
    lambda x: any(m["description"] == "Transferencia bancaria" for m in x)
)
df["transferencia_bancaria"].value_counts()

# %%
# %%
df["acordar_con_el_comprador"] = df[column].apply(
    lambda x: any(m["description"] == "Acordar con el comprador" for m in x)
)
df["acordar_con_el_comprador"].value_counts()


# %%
# Tarjetas
def column_analysis(column: str):
    used_percentage_by_value = df.groupby(column)["used"].mean()
    used_percentage_std_by_value = df.groupby(column)["used"].std()
    z = 1.96
    n = df.groupby(column)["used"].size()

    half = z * used_percentage_std_by_value / np.sqrt(n)
    lower = used_percentage_by_value - half
    upper = used_percentage_by_value + half

    res = pd.DataFrame(
        {
            "n": n,
            "prop_used": used_percentage_by_value,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    ).sort_values("prop_used", ascending=False)

    return res


def plot_column_analysis(res: pd.DataFrame):
    # --- Plot (point estimate with error bars) ---
    x = np.arange(len(res))
    y = res["prop_used"].values
    yerr = np.vstack([y - res["ci_lower"].values, res["ci_upper"].values - y])

    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, res.index, rotation=45, ha="right")
    plt.ylabel("Proportion used (used=1, new=0)")
    plt.title("Proportion of used items by state with 95% CI")
    plt.tight_layout()
    plt.show()


def look_for_likely_to_be_used(res: pd.DataFrame, threshold: float = 0.75):
    return res[(res["prop_used"] > threshold) & (res["ci_lower"] > 0.5)]


def look_for_likely_to_be_new(res: pd.DataFrame, threshold: float = 0.25):
    return res[(res["prop_used"] < threshold) & (res["ci_upper"] < 0.5)]


# %%
res = column_analysis("tarjeta")
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# %%
look_for_likely_to_be_new(res)
# %%
res = column_analysis("efectivo")
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# %%
look_for_likely_to_be_new(res)
# %%
res = column_analysis("mercadopago")
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# %%
look_for_likely_to_be_new(res)
# if it has mercadopago here, it's likely to be new
# %%
res = column_analysis("contra_reembolso")
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# %%
look_for_likely_to_be_new(res)
# %%
res = column_analysis("cheque_certificado")
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# if it has cheque certificado here, it's likely to be used
# %%
res = column_analysis("giro_postal")
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# if it has giro postal here, it's likely to be used
# %%
res = column_analysis("transferencia_bancaria")
plot_column_analysis(res)
# not conclusive
# %%
res = column_analysis("acordar_con_el_comprador")
plot_column_analysis(res)
# not conclusive
# %%
# Adding the payment methods to the dataset seems to be a good idea
