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
column = "attributes"

# %%
df[column].describe()

# %%
df[column].isna().sum() + df[column].isnull().sum()

# %%
df[column].iloc[0]

# %%
df[column].value_counts()
# 88% of the attributes are empty
# %%
