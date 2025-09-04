# %%
import start
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from challenge.new_or_used import build_dataset
from challenge.dataset.preprocess import preprocess_whole_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()

# %%
df["used"] = y_train == "used"
df["used"] = df["used"].astype(int)


# %%
df = preprocess_whole_dataset(df, use_encoders=True)
df.head()

# %%
len(df.columns)

# %%
len(df)
# %%
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)

# %%
# Do a 5 fold cross validation
from sklearn.model_selection import KFold

# %%
# With one fold, train a xgboost classifier
from xgboost import XGBClassifier

# %%
clf = XGBClassifier(enable_categorical=True, class_weight=class_weights)

# %%
y_train = pd.Series([x == "used" for x in y_train])
y_train = y_train.astype(int)
# %%
import pickle

# %%
# Save df
with open("df_train.pkl", "wb") as f:
    pickle.dump(df, f)
# %%
# df = pickle.load(open("df_train.pkl", "rb"))
# %%
# Set to categorical values the columns that are objects
for col in df.columns:
    try:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
    except:
        print(col)
df.dtypes
# %%
kf = KFold(n_splits=5)
from sklearn.metrics import confusion_matrix

for train_index, val_index in kf.split(df):
    clf.fit(df.iloc[train_index], y_train.iloc[train_index])
    print("Score:")
    print(clf.score(df.iloc[val_index], y_train.iloc[val_index]))
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
df_test = pd.DataFrame(X_test)
df_test = preprocess_whole_dataset(df_test, use_encoders=True)
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
with open("df_test.pkl", "wb") as f:
    pickle.dump(df_test, f)
# %%
df_test = pickle.load(open("df_test.pkl", "rb"))
# %%
