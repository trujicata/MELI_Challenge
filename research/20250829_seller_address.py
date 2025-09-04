# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import start  # noqa

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
df["seller_address"].describe()

# %%
df["seller_address"].isna().sum() + df["seller_address"].isnull().sum()

# %%
df["seller_address"].iloc[0]
# %%
df["seller_address_country"] = df["seller_address"].apply(
    lambda x: x["country"]["name"]
)
df["seller_address_state"] = df["seller_address"].apply(lambda x: x["state"]["name"])
df["seller_address_city"] = df["seller_address"].apply(lambda x: x["city"]["name"])
df.drop(columns=["seller_address"], inplace=True)
df.head()
# %%
df["seller_address_country"].value_counts()
# Only 1 country, Argentina
# %%
df.drop(columns=["seller_address_country"], inplace=True)
df.head()
# %%
print(len(df["seller_address_state"].unique()))
df["seller_address_state"].value_counts()
# 25 states
# %%
# for the states, check what percentage of the items are new
# plot it
used_percentage_per_state = df.groupby("seller_address_state")["used"].mean()
used_percentage_per_state
# %%
# But the thing is, each state has a different number of items, so we need to use the standard error
used_percentage_std_per_state = df.groupby("seller_address_state")["used"].std()
z = 1.96
n = df.groupby("seller_address_state")["used"].size()

half = z * used_percentage_std_per_state / np.sqrt(n)
lower = used_percentage_per_state - half
upper = used_percentage_per_state + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_per_state,
        "ci_lower": lower,
        "ci_upper": upper,
    }
).sort_values("prop_used", ascending=False)

# --- Plot (point estimate with error bars) ---
plt.figure(figsize=(10, 6))
x = np.arange(len(res))
y = res["prop_used"].values
yerr = np.vstack([y - res["ci_lower"].values, res["ci_upper"].values - y])

plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
plt.xticks(x, res.index, rotation=45, ha="right")
plt.ylabel("Proportion used (used=1, new=0)")
plt.title("Proportion of used items by state with 95% CI")
plt.tight_layout()
plt.show()

# %%
# I'm looking for states with a percentage near 100% or 0%, with a confidence interval that does not include 0.5
res[(res["prop_used"] > 0.75) & (res["ci_lower"] > 0.5)]
# %%
res[(res["prop_used"] < 0.25) & (res["ci_upper"] < 0.5)]
# %%
# Actually, not much information in the states, let's look at the cities.
# ------------------------------------------------------------
# %%
print(len(df["seller_address_city"].unique()))
df["seller_address_city"].value_counts()
# %%
# There's 3480 cities, let's get the top 100 that have the most items
relevant_cities = df["seller_address_city"].value_counts()[:100].index
relevant_cities
# %%
# for the cities, check what percentage of the items are new
used_percentage_per_city = (
    df[df["seller_address_city"].isin(relevant_cities)]
    .groupby("seller_address_city")["used"]
    .mean()
)
used_percentage_per_city
# %%
# But the thing is, each city has a different number of items, so we need to use the standard error
used_percentage_std_per_city = (
    df[df["seller_address_city"].isin(relevant_cities)]
    .groupby("seller_address_city")["used"]
    .std()
)
z = 1.96
n = (
    df[df["seller_address_city"].isin(relevant_cities)]
    .groupby("seller_address_city")["used"]
    .size()
)

half = z * used_percentage_std_per_city / np.sqrt(n)
lower = used_percentage_per_city - half
upper = used_percentage_per_city + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_per_city,
        "ci_lower": lower,
        "ci_upper": upper,
    }
).sort_values("prop_used", ascending=False)

# --- Plot (point estimate with error bars) ---
plt.figure(figsize=(10, 6))
x = np.arange(len(res))
y = res["prop_used"].values
yerr = np.vstack([y - res["ci_lower"].values, res["ci_upper"].values - y])

plt.figure(figsize=(18, 12))
plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
plt.xticks(x, res.index, rotation=45, ha="right")
plt.ylabel("Proportion used (used=1, new=0)")
plt.title("Proportion of used items by city with 95% CI")
plt.tight_layout()
plt.show()

# %%
# I'm looking for cities with a percentage near 100% or 0%, with a confidence interval that does not include 0.5
a_lot_of_used_cities = list(
    res[(res["prop_used"] > 0.8) & (res["ci_lower"] > 0.5)].index
)
a_lot_of_used_cities
# Bragado, San Andrés de Giles, San Andrés de San Martin, CIUDAD AUTONOMA DE BUENOS AIRES, Coghlan

# %%
a_lot_of_new_cities = list(
    res[(res["prop_used"] < 0.2) & (res["ci_upper"] < 0.5)].index
)
a_lot_of_new_cities
# Trelew, Mataderos, Ciudad Matadero, VILLA SANTA RITA
# %%
