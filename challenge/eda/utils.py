import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def column_analysis(column: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to analize the distribution of the used items in a column
    It returns a dataframe with the number of items in each value, the proportion of used items and the 95% confidence interval
    using the Wilson (it's a binomial distribution) score interval.

    Args:
        column: str, the column to analyze
        df: pd.DataFrame, the dataframe to analyze

    Returns:
        pd.DataFrame, a dataframe with the number of items in each value, the proportion of used items and the confidence interval
    """
    used_percentage_by_value = df.groupby(column)["used"].mean()
    n = df.groupby(column)["used"].size()

    den = 1 + (1.96**2) / n
    center = (used_percentage_by_value + (1.96**2) / (2 * n)) / den
    adj = (
        1.96
        * np.sqrt(
            (used_percentage_by_value * (1 - used_percentage_by_value) / n)
            + (1.96**2) / (4 * n**2)
        )
    ) / den

    wilson_low = center - adj
    wilson_high = center + adj

    res = pd.DataFrame(
        {
            "n": n,
            "prop_used": used_percentage_by_value,
            "ci_lower": wilson_low,
            "ci_upper": wilson_high,
        }
    ).sort_values("prop_used", ascending=False)

    return res


def plot_column_analysis(res: pd.DataFrame) -> None:
    """
    This function is used to plot the distribution of the used items in a column
    It returns a plot with the point estimate and the 95% confidence interval

    Args:
        res: pd.DataFrame, a dataframe with the number of items in each value, the proportion of
        used items and the confidence interval

    Returns:
        None
    """
    x = np.arange(len(res))
    y = res["prop_used"].values
    yerr = np.vstack([y - res["ci_lower"].values, res["ci_upper"].values - y])

    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, res.index, rotation=45, ha="right")
    plt.ylabel("Proportion used (used=1, new=0)")
    plt.title("Proportion of used items by category with 95% CI")
    plt.tight_layout()
    plt.show()


def look_for_likely_to_be_used(
    res: pd.DataFrame, threshold: float = 0.75
) -> pd.DataFrame:
    """
    This function is used to find the values in a column that are likely to be used
    It returns a dataframe with the values that have a proportion of used items greater
    than the threshold and a confidence interval greater than 0.5

    Args:
        res: pd.DataFrame, a dataframe with the number of items in each value, the proportion of
            used items and the confidence interval
        threshold: float, the threshold for the proportion of used items

    Returns:
        pd.DataFrame, a dataframe with the values that have a proportion of used items greater than
        the threshold and a confidence interval greater than 0.5
    """
    return res[(res["prop_used"] > threshold) & (res["ci_lower"] > 0.5)]


def look_for_likely_to_be_new(res: pd.DataFrame, threshold: float = 0.25):
    """
    This function is used to find the values in a column that are likely to be new
    It returns a dataframe with the values that have a proportion of used items less than the threshold and a confidence interval less than 0.5

    Args:
        res: pd.DataFrame, a dataframe with the number of items in each value, the proportion of used items and the confidence interval

    Returns:
        pd.DataFrame, a dataframe with the values that have a proportion of used items less than the threshold and a confidence interval
        less than 0.5
    """
    return res[(res["prop_used"] < threshold) & (res["ci_upper"] < 0.5)]
