# %%
import pandas as pd
import start  # noqa

from challenge.new_or_used import build_dataset

# %%
X_train, y_train, X_test, y_test = build_dataset()
df = pd.DataFrame(X_train)
df.head()

# %%
df.columns

# %%
df["warranty"].describe()
# text, the seller describes the warranty
# %%
df["sub_status"].value_counts()
# empty, suspended, expired, deleted
# %%
df["condition"].value_counts()
# new, used
# %%
df["deal_ids"].value_counts()
# code of the deal, not much information. Mostly empty.
# %%
df["variations"].value_counts()
# If there is variations, gives the info on every variation.
# %%
df["seller_address"].value_counts()
# dict with country, state, city
# %%
df["shipping"].value_counts()
# dict with if it has local_pickup, methods (?), tags (a list), free_shipping, mode, and dimensions (very inconsistent)
# %%
df["non_mercado_pago_payment_methods"].value_counts()
# list of dictionaries with payment methods and their info
# %%
df["seller_id"].describe()
# float64, the id of the seller
# %%
df["site_id"].value_counts()
# MLA only
# %%
df["listing_type_id"].value_counts()
# Some kind of type of listing, that can be free, bronze, silver, gold, etc.
# %%
df["price"].describe()
# float64, the price of the item
# %%
df["attributes"].value_counts().index[1]
# list of dictionaries of attributes and information about them: value_id, attribute_group_id, name, value_name, attribute_group_name, id
# %%
df["buying_mode"].value_counts()
# buy_it_now (mostly), auction, or undefined
# %%
df["tags"].value_counts()
# list of strings. Not sure what they are.
# %%
df["listing_source"].iloc[0]
# empty string for everything
# %%
df["parent_item_id"].value_counts()
# string with some other ID! We could make some relation between the items
# %%
df["coverage_areas"].value_counts()
# Empty list for everything
# %%
df["category_id"].describe()
# 10491 different categories. The one most common has 4139 items (too small to be useful)
# %%
df["descriptions"].describe()
# A list of dictionaries with a code like MLA453243... and nothing more
# %%
df["last_updated"].describe()
# datetime64[ns]
# %%
df["international_delivery_mode"].value_counts()
# all in none
# %%
df["pictures"].iloc[0]
# Can't access the pictures! Only can see how many there are, the size, the max size and the quality
# %%
df["id"].describe()
# unique ID, strings
# %%
# df["official_store_id"].describe()
df["official_store_id"].unique()
# float, the id of the official store. Or nan, or in between 745 stores
# %%
df["differential_pricing"].value_counts()
# empty for everything
# %%
df["accepts_mercadopago"].value_counts()
# True (mostly) or false
# %%
df["original_price"].describe()
# float64, the original price of the item
# %%
df["currency_id"].value_counts()
# ARS (mostly) and USD
#
# %%
df["thumbnail"].iloc[0]
# Not accessible
# %%
df["title"].describe()
# string, the title of the item
# %%
df["automatic_relist"].value_counts()
# False (mostly) or true
# %%
df["date_created"].describe()
# datetime64[ns]
# %%
df["secure_thumbnail"].iloc[0]
# str, access denied
# %%
df["stop_time"].iloc[0]
# np.int64
# %%
df["status"].value_counts()
# active (mostly), paused, closed, not_yet_active
# %%
df["video_id"].value_counts()
# None for most of them, if not there's a string with the id
# %%
df["catalog_product_id"].unique()
# The id of the product. 7 ids and the rest are nan
# %%
df["subtitle"].value_counts()
# empty for all of them
# %%
df["initial_quantity"].value_counts()
# int64, the initial quantity of the item. 1 for most of them
# %%
df["start_time"].iloc[0]
# np.int64
# %%
df["permalink"].iloc[0]
# string, the permalink of the item
# %%
df["sold_quantity"].value_counts()
# int. mostly 0
# %%
df["available_quantity"].value_counts()
# int64, the available quantity of the item. 1 for most of them
