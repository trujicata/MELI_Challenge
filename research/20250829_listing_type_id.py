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
column = "listing_type_id"

# %%
df[column].describe()

# %%
df[column].isna().sum() + df[column].isnull().sum()

# %%
df[column].iloc[2]

# %%
df[column].value_counts()
# %%
listing_types = list(df[column].value_counts().index)
listing_types


# %%
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
res = column_analysis(column)
res
# %%
plot_column_analysis(res)
# %%
look_for_likely_to_be_used(res)
# free is the most likely to be used
# %%
look_for_likely_to_be_new(res)
# silver, gold, gold_special, and gold_pro are the most likely to be new
# %%
