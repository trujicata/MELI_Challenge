# %%
import start  # noqa
from challenge.new_or_used import build_dataset
import pandas as pd
import numpy as np

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()
# %%
df["used"] = [y == "used" for y in y_train]
df["used"] = df["used"].astype(int)
# %%
# Convert to time
df["start_time"] = pd.to_datetime(df["start_time"])
df["stop_time"] = pd.to_datetime(df["stop_time"])
# %%
df["total_time"] = df["stop_time"] - df["start_time"]
df["total_time"].describe()
# %%
df["start_time"].max()
# %%
df["start_time"].min()
# %%
df["total_time"].min()
# %%
df["total_time"].max()
# %%
df["total_time_seconds"] = df["total_time"].dt.total_seconds()
df["total_time_seconds"].describe()
# %%
df["total_time_seconds"].min()
# %%
df["total_time_seconds"].max()
# %%
df["total_time_seconds"].isnull().sum()
# %%
spearman_correlation = df[["used", "total_time_seconds"]].corr(method="spearman")
spearman_correlation
# %%
pearson_correlation = df[["used", "total_time_seconds"]].corr(method="pearson")
pearson_correlation
# %%
kendall_correlation = df[["used", "total_time_seconds"]].corr(method="kendall")
kendall_correlation
# %%
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# %%
scaler = MinMaxScaler()
scaler.fit(df[["total_time_seconds"]])
df["total_time_seconds_scaled"] = scaler.transform(df[["total_time_seconds"]]).flatten()
sns.histplot(df["total_time_seconds_scaled"])
plt.yscale("log")

plt.show()
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(df.loc[df["used"] == 1]["total_time_seconds"], ax=axs[0])
sns.histplot(df.loc[df["used"] == 0]["total_time_seconds"], ax=axs[1])
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.show()
# %%
df.loc[df["used"] == 1]["total_time_seconds"].describe()
# %%
df.loc[df["used"] == 0]["total_time_seconds"].describe()
# %%
