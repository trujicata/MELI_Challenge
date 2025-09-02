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
column = "shipping"

# %%
df[column].describe()

# %%
df[column].isna().sum() + df[column].isnull().sum()

# %%
df[column].iloc[0]
# %%
df["shipping_local_pick_up"] = df[column].apply(lambda x: x["local_pick_up"])
df["shipping_local_pick_up"].value_counts()
# %%
(
    df["shipping_local_pick_up"].isna().sum()
    + df["shipping_local_pick_up"].isnull().sum()
)  # 0
# %%
df["shipping_tags"] = df[column].apply(lambda x: x["tags"])
df["shipping_tags"].value_counts()
# Almost all the data is the same
# %%
df["shipping_tags"].isna().sum() + df["shipping_tags"].isnull().sum()

# %%
df["shipping_free_shipping"] = df[column].apply(lambda x: x["free_shipping"])
df["shipping_free_shipping"].value_counts()
# %%
df["shipping_free_shipping"].isna().sum()
+df["shipping_free_shipping"].isnull().sum()
# %%
df["shipping_mode"] = df[column].apply(lambda x: x["mode"])
df["shipping_mode"].value_counts()
# %%
df["shipping_mode"].isna().sum() + df["shipping_mode"].isnull().sum()
# %%# %%
df["shipping_dimensions"] = df[column].apply(lambda x: x["dimensions"])
df["shipping_dimensions"].value_counts()
# %%
df["shipping_dimensions"].isna().sum()
# 89978 , almost 100% of the data


# %%
def check_methods(value: dict):
    if "methods" in value.keys():
        return value["methods"]
    elif "free_methods" in value.keys():
        return value["free_methods"]
    else:
        return None


# %%
df[column].apply(check_methods).value_counts()
# Almost all the data is the same
# %%
# Let's focus then on: shipping_local_pick_up, shipping_free_shipping, shipping_mode

# %%
# Shipping_local_pick_up

# %%
used_percentage_id_local_pickup = df.groupby("shipping_local_pick_up")["used"].mean()
used_percentage_id_local_pickup
# %%
# But the thing is, each state has a different number of items, so we need to use the standard error
used_percentage_std_per_state = df.groupby("shipping_local_pick_up")["used"].std()
z = 1.96
n = df.groupby("shipping_local_pick_up")["used"].size()

half = z * used_percentage_std_per_state / np.sqrt(n)
lower = used_percentage_id_local_pickup - half
upper = used_percentage_id_local_pickup + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_id_local_pickup,
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
plt.title("Proportion of used items by state with 95% CI")
plt.tight_layout()
plt.show()

# %%
# I'm looking for states with a percentage near 100% or 0%, with a confidence interval that does not include 0.5
res[(res["prop_used"] > 0.75) & (res["ci_lower"] > 0.5)]
# %%
res[(res["prop_used"] < 0.25) & (res["ci_upper"] < 0.5)]

# %%
# shipping_free_shipping
used_percentage_id_free_shipping = df.groupby("shipping_free_shipping")["used"].mean()
used_percentage_id_free_shipping
# %%
used_percentage_std_per_state = df.groupby("shipping_free_shipping")["used"].std()
z = 1.96
n = df.groupby("shipping_free_shipping")["used"].size()

half = z * used_percentage_std_per_state / np.sqrt(n)
lower = used_percentage_id_free_shipping - half
upper = used_percentage_id_free_shipping + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_id_free_shipping,
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
plt.title("Proportion of used items by free shipping with 95% CI")
plt.tight_layout()
plt.show()

# %%
res[(res["prop_used"] > 0.75) & (res["ci_lower"] > 0.5)]
# %%
res[(res["prop_used"] < 0.25) & (res["ci_upper"] < 0.5)]
# If we have a free shipping, we have a higher chance of having a used item
# %%
# shipping_mode
used_percentage_id_mode = df.groupby("shipping_mode")["used"].mean()
used_percentage_id_mode
# %%
used_percentage_std_per_state = df.groupby("shipping_mode")["used"].std()
z = 1.96
n = df.groupby("shipping_mode")["used"].size()

half = z * used_percentage_std_per_state / np.sqrt(n)
lower = used_percentage_id_mode - half
upper = used_percentage_id_mode + half

# %%

res = pd.DataFrame(
    {
        "n": n,
        "prop_used": used_percentage_id_mode,
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
plt.title("Proportion of used items by shipping mode with 95% CI")
plt.tight_layout()
plt.show()

# %%
res[(res["prop_used"] > 0.75) & (res["ci_lower"] > 0.5)]
# %%
res[(res["prop_used"] < 0.25) & (res["ci_upper"] < 0.5)]
# me1 is more likely to be new. The rest of the modes are not conclusive
# %%
