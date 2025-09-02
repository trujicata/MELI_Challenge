import datetime
import json
import pickle

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from challenge.dataset.preprocess import preprocess_whole_dataset
from challenge.new_or_used import build_dataset


def find_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: np.ndarray,
    n_trials: int = 20,
):
    """
    Find the best parameters for the XGBoost model.
    It uses the optuna library to find the best parameters, looking for maximizing
    the average f1 score for the validation set

    Args:
        X_train: The training data.
        y_train: The training labels.
        class_weights: The class weights.
        n_trials: The number of trials.
    """

    def objective(trial):
        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, 100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.02, step=0.005
            ),
            "max_depth": trial.suggest_int("max_depth", 10, 20, 1),
            "subsample": trial.suggest_float("subsample", 0.6, 1, step=0.05),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.8, 1, step=0.05
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10, 1),
            "gamma": trial.suggest_float("gamma", 0.6, 1, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 1, step=0.1),
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

        f1s = []

        for train_index, val_index in kf.split(X_train):
            clf.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            y_pred = clf.predict(X_train.iloc[val_index])
            f1 = f1_score(y_train.iloc[val_index], y_pred)
            f1s.append(f1)

        return np.mean(f1s)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def main():
    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    X_train, y_train, X_test, y_test = build_dataset()
    X_train = preprocess_whole_dataset(X_train)
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].astype("category")
    y_train = pd.Series([y == "used" for y in y_train])
    y_train = y_train.astype(int)

    X_test = preprocess_whole_dataset(X_test)
    for col in X_test.columns:
        if X_test[col].dtype == "object":
            X_test[col] = X_test[col].astype("category")
    y_test = pd.Series([y == "used" for y in y_test])
    y_test = y_test.astype(int)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    best_params, best_value = find_best_params(
        X_train, y_train, class_weights, n_trials=20
    )
    print("For the parameter optimization, the best f1 average was: ", best_value)
    print("The best parameters were: ", best_params)

    clf = XGBClassifier(**best_params)
    clf.fit(X_train, y_train)

    train_recall = recall_score(y_train, clf.predict(X_train))
    test_recall = recall_score(y_test, clf.predict(X_test))
    train_precision = precision_score(y_train, clf.predict(X_train))
    test_precision = precision_score(y_test, clf.predict(X_test))
    train_f1 = f1_score(y_train, clf.predict(X_train))
    test_f1 = f1_score(y_test, clf.predict(X_test))
    train_confusion_matrix = confusion_matrix(y_train, clf.predict(X_train))
    test_confusion_matrix = confusion_matrix(y_test, clf.predict(X_test))

    results = {
        "id": id,
        "best_params": best_params,
        "train_recall": train_recall,
        "best_average_f1_for_validation": best_value,
        "train_precision": train_precision,
        "train_f1": train_f1,
        "train_confusion_matrix": train_confusion_matrix,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_f1": test_f1,
        "test_confusion_matrix": test_confusion_matrix,
    }

    # Save the model and save best_params
    with open(
        f"challenge/models/classic_xgboost/params/best_params_{id}_{test_f1}.json",
        "w",
    ) as f:
        json.dump(results, f)
    with open(
        f"challenge/models/classic_xgboost/weights/model_{id}_{test_f1}.pkl",
        "wb",
    ) as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
