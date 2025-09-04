# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import start  # noqa

from challenge.new_or_used import build_dataset

# %%
df = build_dataset()
df.head()

# %%
column = "tags"

# %%
df[column].describe()

# %%
df[column].isna().sum() + df[column].isnull().sum()

# %%
df[column].iloc[2]

# %%
df[column].value_counts()

# %%
possible_tags = list(df[column].value_counts().index)
all_tags = []
for tag in possible_tags:
    all_tags.extend(tag)
possible_tags = list(set(all_tags))
possible_tags
# %%
for tag in possible_tags:
    print(tag)
    df[f"tag_{tag}"] = df[column].apply(lambda x: tag in x)

# %%
df["empty_tags"] = df[column].apply(lambda x: len(x) == 0)
df["empty_tags"].value_counts()
# %%
columns_to_analyze = ["empty_tags"] + [f"tag_{tag}" for tag in possible_tags]
columns_to_analyze


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
print(possible_tags[0])
res = column_analysis(f"tag_{possible_tags[0]}")
res
# Only 723 with dragged_visits
# %%
plot_column_analysis(res)
# Inconclusive
# %%
print(possible_tags[1])
res = column_analysis(f"tag_{possible_tags[1]}")
res
# %%
plot_column_analysis(res)
# Inconclusive
# %%
print(possible_tags[2])
res = column_analysis(f"tag_{possible_tags[2]}")
res
# Only 13 poor quality thumbnails!
# %%
print(possible_tags[3])
res = column_analysis(f"tag_{possible_tags[3]}")
res
# Only 1537 with tag_good_quality_thumbnail
# %%
print(possible_tags[4])
res = column_analysis(f"tag_{possible_tags[4]}")
res  # Only 259 with tag_free_relist
