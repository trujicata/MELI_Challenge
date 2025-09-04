# %%
import start
import pandas as pd
import numpy as np
import optuna
import pickle

from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_class_weight
from challenge.new_or_used import build_dataset
from challenge.dataset.preprocess import preprocess_whole_dataset

# %%
_, y_train, _, y_test = build_dataset()

# %%
df = pickle.load(open("df_train.pkl", "rb"))
# %%
for col in df.columns:
    try:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
    except:
        print(col)
df.dtypes
# %%
from sklearn.metrics import confusion_matrix, accuracy_score

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
y_train = pd.Series([x == "used" for x in y_train])
y_train = y_train.astype(int)


def objective(trial):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02),
        "max_depth": trial.suggest_int("max_depth", 10, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.6, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 1),
        "random_state": 42,
        "scale_pos_weight": class_weights[1],
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
        "enable_categorical": True,
    }
    clf = XGBClassifier(**param)
    kf = KFold(n_splits=5)

    accuracies = []

    for train_index, val_index in kf.split(df):
        clf.fit(df.iloc[train_index], y_train.iloc[train_index])
        y_pred = clf.predict(df.iloc[val_index])
        accuracy = accuracy_score(y_train.iloc[val_index], y_pred)
        accuracies.append(accuracy)

    return np.mean(accuracies)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
# %%
best_params = study.best_params
best_params
# %%
best_params = {
    "n_estimators": 300,
    "learning_rate": 0.006,
    "max_depth": 14,
    "subsample": 0.95,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "gamma": 0.9,
    "reg_alpha": 0.7000000000000001,
    "reg_lambda": 0.4,
    "grow_policy": "lossguide",
    "tree_method": "hist",
}
# %%
len(df.columns)

# %%
len(df)
# %%
clf = XGBClassifier(enable_categorical=True, class_weight=class_weights, **best_params)
# %%
clf.fit(df, y_train)
clf.score(df, y_train)

# %%
confusion_matrix(y_train, clf.predict(df))
# %%
# Match feature names with feature importances
feature_importance_df = pd.DataFrame(
    {"feature": df.columns, "importance": clf.feature_importances_}
)

# Sort by importance (descending)
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)

# Display top features
feature_importance_df
# %%
df_test = pickle.load(open("df_test.pkl", "rb"))
df_test.head()
# %%
df_test.dtypes
# %%
for col in df_test.columns:
    if df_test[col].dtype == "object":
        df_test[col] = df_test[col].astype("category")
# %%
df_test.dtypes
# %%
df_test.head()
# %%
y_test = pd.Series([x == "used" for x in y_test])
y_test = y_test.astype(int)
# %%
clf.predict(df_test)
# %%
confusion_matrix(y_test, clf.predict(df_test))
# %%
clf.score(df_test, y_test)
# %%
# Save model
import pickle

with open("clf.pkl", "wb") as f:
    pickle.dump(clf, f)
# %%
clf = pickle.load(open("clf.pkl", "rb"))
# %%
clf.score(df_test, y_test)
# %%
