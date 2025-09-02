# %%
import start  # noqa

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

from challenge.new_or_used import build_dataset
from challenge.dataset.preprocess import preprocess_whole_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()

# %%
X_train = preprocess_whole_dataset(df)
for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = X_train[col].astype("category")
X_train.dtypes

# %%
y_train = pd.Series([y == "used" for y in y_train])
y_train = y_train.astype(int)


# %%
study = optuna.create_study(direction="maximize")
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(y_train),
    y=y_train,
)


def objective(trial):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.005),
        "max_depth": trial.suggest_int("max_depth", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.9, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.9, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.6, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 1),
        "random_state": 42,
        "scale_pos_weight": class_weights[1],
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "tree_method": "hist",
        "enable_categorical": True,
    }
    clf = XGBClassifier(**param, early_stopping_rounds=50)
    kf = KFold(n_splits=5)

    accuracies = []

    for train_index, val_index in kf.split(X_train):
        clf.fit(
            X_train.iloc[train_index],
            y_train.iloc[train_index],
            eval_set=[(X_train.iloc[val_index], y_train.iloc[val_index])],
            verbose=False,
        )
        y_pred = clf.predict(X_train.iloc[val_index])
        accuracy = accuracy_score(y_train.iloc[val_index], y_pred)
        accuracies.append(accuracy)

    return np.mean(accuracies)


# %%
study.optimize(objective, n_trials=20)
best_params = study.best_params
print(f"Best params: {best_params}")

# %%
study.best_params

# %%
study.best_value

# %%
X_test = preprocess_whole_dataset(pd.DataFrame(X_test))
for col in X_test.columns:
    if X_test[col].dtype == "object":
        X_test[col] = X_test[col].astype("category")

# %%
y_test = pd.Series([y == "used" for y in y_test])
y_test = y_test.astype(int)

# %%
clf = XGBClassifier(**best_params, enable_categorical=True, early_stopping_rounds=50)
clf.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)
# %%
accuracy_score(y_test, clf.predict(X_test))
# %%
recall_score(y_test, clf.predict(X_test))

# %%
precision_score(y_test, clf.predict(X_test))

# %%
f1_score(y_test, clf.predict(X_test))
# %%
confusion_matrix(y_test, clf.predict(X_test))
# %%

# %%
# Let's study, in training, why do the model have this performance.
y_pred = clf.predict(X_train)

y_pred
# %%
X_train["y_pred"] = y_pred
X_train["y_true"] = y_train
# %%
misses = X_train.loc[X_train["y_pred"] != X_train["y_true"]]
misses.head()
# %%
# Match feature names with feature importances
feature_importance_df = pd.DataFrame(
    {"feature": X_train.columns[:-2], "importance": clf.feature_importances_}
)

# Sort by importance (descending)
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)

# Display top features
feature_importance_df

# %%
misses["initial_quantity"].value_counts()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(misses["initial_quantity"])
plt.xscale("log")
plt.show()
# %%
sns.histplot(misses["initial_quantity"])
plt.show()
# %%
fig, ax = plt.subplots(1, 2, figsize=(18, 5))

sns.histplot(
    misses.loc[misses["y_true"] == 1]["initial_quantity"],
    ax=ax[0],
    bins=[0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000],
)
ax[0].set_title("Histogram of the initial quantity for used items")
ax[0].set_yscale("log")
ax[0].set_xscale("log")

sns.histplot(
    misses.loc[misses["y_true"] == 0]["initial_quantity"],
    ax=ax[1],
    bins=[0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000],
)
ax[1].set_title("Histogram of the initial quantity for new items")
ax[1].set_yscale("log")
ax[1].set_xscale("log")
plt.tight_layout()
plt.show()
# %%
misses["listing_type_id"].value_counts()


# %%
def column_analysis(column: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to analize the distribution of the used items in a column
    It returns a dataframe with the number of items in each value, the proportion of used items and the 95% confidence interval
    using the Wilson score interval

    Args:
        column: str, the column to analyze
        df: pd.DataFrame, the dataframe to analyze

    Returns:
        pd.DataFrame, a dataframe with the number of items in each value, the proportion of used items and the confidence interval
    """
    used_percentage_by_value = df.groupby(column)["y_true"].mean()
    n = df.groupby(column)["y_true"].size()

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


res = column_analysis("listing_type_id", misses)
res

# %%
