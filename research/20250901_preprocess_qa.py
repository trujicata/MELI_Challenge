# %%
import pandas as pd
import start
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from challenge.dataset.preprocess import preprocess_whole_dataset
from challenge.new_or_used import build_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()

# %%
y_train = pd.Series([y_t == "used" for y_t in y_train])
y_train = y_train.astype(int)

# %%
df = preprocess_whole_dataset(df)
df.head()

# %%
len(df.columns)

# %%
len(df)
# %%
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category")
# %%
df.dtypes
# %%
clf = XGBClassifier(enable_categorical=True)

clf.fit(df, y_train)
# %%
clf.score(df, y_train)
# %%
feature_importance_df = pd.DataFrame(
    {"feature": df.columns, "importance": clf.feature_importances_}
)

# Sort by importance (descending)
feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)

# Display top features
feature_importance_df

# %%
decreasing_order_feat_imp = feature_importance_df["feature"].tolist()

# %%
feature = decreasing_order_feat_imp[-1]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-2]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-3]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-4]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-5]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-6]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-7]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-8]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-9]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-10]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-11]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-12]
df[feature].value_counts()

# %%
feature = decreasing_order_feat_imp[-13]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-14]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-15]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-16]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-17]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-18]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-19]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-20]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-21]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-22]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-23]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-24]
df[feature].value_counts()
# %%
feature = decreasing_order_feat_imp[-25]
df[feature].value_counts()
# %%
