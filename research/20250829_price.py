# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import start  # noqa
from argendolar import Argendolar, TipoDivisas

from challenge.new_or_used import build_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()
df = pd.DataFrame(X_train)
df.head()

# %%
df["price"].describe()
# %%
df["price"].isnull().sum()
# %%
# Let's check the currency for the base price
# For that we need the "currency_id" column and the "last_updated" column
df["currency_id"].value_counts()

# %%
df["last_updated"].min()
# %%
argendolar = Argendolar()
dolar_oficial = argendolar.get_dolar_historia_completa(tipo=TipoDivisas.OFICIAL)


# %%
def transform_into_USD(price, currency_id, last_updated):
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
        return price * dolar_price


df["price_usd"] = df.apply(
    lambda row: transform_into_USD(
        row["price"], row["currency_id"], row["last_updated"]
    ),
    axis=1,
)
df["price_usd"].describe()
# %%
df["price"].describe()
# %%
# do a sns histogram of the base price in USD
bins = [0, 1, 10, 250, 1000, 10000, 20000]
sns.histplot(df["price_usd"], bins=bins)
plt.xscale("log")
plt.ylim(0, 50000)
plt.title("Base price in USD")
plt.show()
# %%
# do a sns histogram of the base price in USD for used items and new items
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(df[df["used"] == 1]["price_usd"], bins=bins, ax=axs[0])
axs[0].set_xscale("log")
axs[0].set_ylim(0, 28000)
axs[0].set_title("Base price in USD for used items")

sns.histplot(df[df["used"] == 0]["price_usd"], bins=bins, ax=axs[1])
axs[1].set_xscale("log")
axs[1].set_ylim(0, 28000)
axs[1].set_title("Base price in USD for new items")
plt.tight_layout()
plt.show()

# %%
df[df["used"] == 1]["price_usd"].describe()
# %%
# do a sns histogram of the base price in USD for new items
df[df["used"] == 0]["price_usd"].describe()
# %%
# The used items have a slightly higher mean price, but a bigger std too
# There's a bigger concentration of prices more than 1000 USD for new items
# %%
# Look for Pearson correlation between the "used" column and the "price_usd" column
pearson_corr = df[["used", "price_usd"]].corr(method="pearson")
pearson_corr  # 0.000931
# %%
# Look for Spearman correlation between the "used" column and the "price_usd" column
spearman_corr = df[["used", "price_usd"]].corr(method="spearman")
spearman_corr  # -0.2025
# %%
# Look for Kendall correlation between the "used" column and the "price_usd" column
kendall_corr = df[["used", "price_usd"]].corr(method="kendall")
kendall_corr  # -0.1659

# %%
# the spearman makes more sense, because it's a monotonic relationship. The "newer" it is, the higher the price
# Same as base price!!
