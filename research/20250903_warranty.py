# %%
import start  # noqa
from challenge.new_or_used import build_dataset
import pandas as pd
import numpy as np
import torch
from challenge.dataset.utils import typical_string_processing
from challenge.dataset.warranty import warranty_string_processing

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()
# %%
train_warranties = df["warranty"]
train_warranties
# %%
train_warranties.describe()
# %%
train_warranties.value_counts()
# %%
train_warranties.value_counts(normalize=True)
# %%
y_train = [y == "used" for y in y_train]
y_test = [y == "used" for y in y_test]
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)
# %%
df_test = pd.DataFrame(X_test)
test_warranties = df_test["warranty"]
test_warranties
# %%
test_warranties.describe()
# %%
test_warranties.value_counts()
# %%
warranty_df_train_df = pd.DataFrame({"warranty": train_warranties, "y": y_train})
warranty_df_test_df = pd.DataFrame({"warranty": test_warranties, "y": y_test})
# %%
warranty_df_train_df.head()

# %%
warranty_df_test_df["processed_warranty"] = warranty_df_test_df["warranty"].apply(
    warranty_string_processing
)
warranty_df_train_df["processed_warranty"] = warranty_df_train_df["warranty"].apply(
    warranty_string_processing
)
# %%
warranty_df_train_df["processed_warranty"].value_counts()
# %%
warranty_df_train_df.loc[warranty_df_train_df["processed_warranty"] == "otro"][
    "warranty"
].value_counts()[:25]


# %%
def warranty_mis_calificaciones(x):
    first_check = warranty_string_processing(x)
    if first_check == "otro":
        x = typical_string_processing(x)
        possible_mis_calificaciones = [
            "mis calificaciones",
            "mi calificacion",
            "calificacion",
            "reputacion",
        ]
        mis_alguna_calificacion = False
        for calificacion in possible_mis_calificaciones:
            if calificacion in x:
                mis_alguna_calificacion = True
                break
        return mis_alguna_calificacion
    else:
        return False


# %%
warranty_df_train_df["mis_calificaciones"] = warranty_df_train_df["warranty"].apply(
    warranty_mis_calificaciones
)
# %%
warranty_df_train_df["mis_calificaciones"].value_counts()

# %%
warranty_df_train_df.loc[
    ~warranty_df_train_df["mis_calificaciones"]
    & warranty_df_train_df["processed_warranty"]
    == "otro"
]["warranty"].value_counts()[:25]

# %%
