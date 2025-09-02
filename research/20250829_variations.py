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
column = "variations"

# %%
df[column].describe()

# %%
df[column].isna().sum() + df[column].isnull().sum()

# %%
df[column].iloc[2]

# %%
df[column].value_counts()
# 92 % of the variations are empty
