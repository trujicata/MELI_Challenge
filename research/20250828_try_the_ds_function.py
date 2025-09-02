# %%
import start  # noqa
from challenge.new_or_used import build_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
print(X_train[0])
print(y_train[0])

# %%
one_X = X_train[0]
one_X.keys()
# %%
import pandas as pd

df = pd.DataFrame(X_train)
df.head()

# %%
print(len(df.columns))
df.columns
# %%
df["seller_address"]
# %%
df["sub_status"]
# %%
df["shipping"][0]
# %%
# Get the type of every column. Count the number of columns for each type.
type_counts = {}
for col in df.columns:
    type_counts[type(df[col].iloc[0])] = type_counts.get(type(df[col].iloc[0]), 0) + 1
type_counts

# %%
# For every column that has dictionaries, print the keys and values
for col in df.columns:
    if isinstance(df[col].iloc[0], dict):
        print(col)
        print(df[col].iloc[0])
        print(df[col].iloc[1])
        print(df[col].iloc[2])
        print(df[col].iloc[3])
        print(df[col].iloc[4])
# %%
# For every column that has lists, print the first 5 elements
for col in df.columns:
    if isinstance(df[col].iloc[0], list):
        print("--------------------------------")
        print(col)
        print(df[col].iloc[0][:5])
# %%
# For every column that has strings, print the first 5 element
for col in df.columns:
    if isinstance(df[col].iloc[0], str):
        print("--------------------------------")
        print(col)
        print(df[col].iloc[0])
        print(df[col].iloc[1])
        print(df[col].iloc[2])
        print(df[col].iloc[3])
        print(df[col].iloc[4])
