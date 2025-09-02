# %%
import start  # noqa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from challenge.new_or_used import build_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()

# %%
df["used"] = [i == "used" for i in y_train]
df["used"] = df["used"].astype(int)
df["used"].value_counts()
# %%
df.head()

# %%
df["warranty"].describe()

# %%
df["warranty"].value_counts()[:100]

# %%
# Search for NaN, None, and empty strings
df["warranty"].isnull().sum() + df["warranty"].apply(lambda x: x == "").sum()


# 54757 in total, which is 61% of the data
# %%
# Fill with "unknown"
df["warranty"] = df["warranty"].fillna("unknown")
df["warranty"] = df["warranty"].apply(lambda x: "unknown" if x == "" else x)
df["warranty"].value_counts()


# %%
def typical_string_processing(x):
    tildes = "áéíóú"
    no_tildes = "aeiou"
    x = x.lower()
    x = x.replace(".", "")
    x = x.strip()
    for i in range(len(tildes)):
        x = x.replace(tildes[i], no_tildes[i])
    return x


def find_warranty_magnitude(x):
    if "año" in x or "ano" in x.split(" ") or "anos" in x.split(" "):
        return "años"
    elif "mes" in x:
        return "meses"
    elif "semana" in x:
        return "semanas"
    elif "dia" in x:
        # Find out the number of days
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
        return "dias"
    else:
        return x


def warranty_de_fabricacion(x):
    if "fabrica" in x or "origen" in x:
        return "de fabrica"
    else:
        return x


def yes_no_warranty(x):
    if "si" in x.split(" "):
        return "si"
    elif "sin garantia" in x:
        return "no"
    else:
        return x


def warranty_estado(x):
    if x not in [
        "años",
        "meses",
        "semanas",
        "dias",
        "de fabrica",
        "unknown",
        "si",
        "no",
    ]:
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
            return "otro"
    else:
        return x


def warranty_processing(x):
    if isinstance(x, str) and x != "unknown":
        x = typical_string_processing(x)
        x = find_warranty_magnitude(x)
        x = warranty_de_fabricacion(x)
        x = yes_no_warranty(x)
        x = warranty_estado(x)
    return x


# %%
df["warranty"].apply(warranty_processing).value_counts()
# %%
df["warranty_processed"] = df["warranty"].apply(warranty_processing)
df.loc[df["warranty_processed"] == "otro"]["warranty"].value_counts()[:30]


# %%
used_percentage_warranty = df.groupby("warranty_processed")["used"].mean()
used_percentage_warranty
# %%
# But the thing is, each warranty has a different number of items, so we need to use the standard error
used_std_per_warranty = df.groupby("warranty_processed")["used"].std()
z = 1.96
n = df.groupby("warranty_processed")["used"].size()

half = z * used_std_per_warranty / np.sqrt(n)
lower = used_percentage_warranty - half
upper = used_percentage_warranty + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_warranty,
        "ci_lower": lower,
        "ci_upper": upper,
    }
).sort_values("prop_used", ascending=False)

# --- Plot (point estimate with error bars) ---
x = np.arange(len(res))
y = res["prop_used"].values
yerr = np.vstack([y - res["ci_lower"].values, res["ci_upper"].values - y])

plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
plt.xticks(x, res.index, rotation=45, ha="right")
plt.ylabel("Proportion used (used=1, new=0)")
plt.title("Proportion of used items by warranty with 95% CI")
plt.tight_layout()
plt.show()

# %%
# I'm looking for warranties with a percentage near 100% or 0%, with a confidence interval that does not include 0.5
res[(res["prop_used"] > 0.8) & (res["ci_lower"] > 0.5)]
# %%

a_lot_of_used_warranties = list(
    res[(res["prop_used"] > 0.8) & (res["ci_lower"] > 0.5)].index
)
a_lot_of_used_warranties
# "usado"
# %%
res[(res["prop_used"] < 0.1) & (res["ci_upper"] < 0.5)]

# %%
a_lot_of_new_warranties = list(
    res[(res["prop_used"] < 0.1) & (res["ci_upper"] < 0.5)].index
)
a_lot_of_new_warranties

# 'a nuevo', 'años', 'meses', 'de fabrica'

# %%
