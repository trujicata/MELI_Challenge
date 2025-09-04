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
df["deal_ids"].value_counts()
# %%
df["deal_ids"].isnull().sum()


# %%
def deal_ids_processing(x):
    if len(x) > 0:
        return "ID"
    else:
        return "unknown"


df["deal_ids"].apply(deal_ids_processing).value_counts()
# %%

df["deal_ids"] = df["deal_ids"].apply(deal_ids_processing)
df["deal_ids"].value_counts()

# %%
used_percentage_substatus = df.groupby("deal_ids")["used"].mean()
used_percentage_substatus
# %%
# But the thing is, each substatus has a different number of items, so we need to use the standard error
used_std_per_substatus = df.groupby("deal_ids")["used"].std()
z = 1.96
n = df.groupby("deal_ids")["used"].size()

half = z * used_std_per_substatus / np.sqrt(n)
lower = used_percentage_substatus - half
upper = used_percentage_substatus + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_substatus,
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
plt.title("Proportion of used items by substatus with 95% CI")
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

# %%
# If there's an ID, it's probably new!

# %%
