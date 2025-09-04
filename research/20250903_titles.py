# %%
import start  # noqa
from challenge.new_or_used import build_dataset
import pandas as pd
import numpy as np
import torch

# %%
X_train, y_train, X_test, y_test = build_dataset()

# %%
df = pd.DataFrame(X_train)
df.head()
# %%
train_titles = df["title"]
train_titles
# %%
train_titles.describe()
# %%
train_titles.value_counts()
# %%
train_titles.value_counts(normalize=True)
# %%
y_train = [y == "used" for y in y_train]
y_test = [y == "used" for y in y_test]
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)
# %%
df_test = pd.DataFrame(X_test)
test_titles = df_test["title"]
test_titles
# %%
test_titles.describe()
# %%
test_titles.value_counts()
# %%
title_df_train_df = pd.DataFrame({"title": train_titles, "y": y_train})
title_df_test_df = pd.DataFrame({"title": test_titles, "y": y_test})
# %%
title_df_train_df.head()
# %%
title_df_test_df.head()
# %%
# Common words for sex shop
from challenge.dataset.utils import typical_string_processing

sex_shop_one_word_words = [
    "consolador",
    "vibrador",
    "anal",
    "pene",
    "peneano",
    "masajes",
]
sex_shop_two_word_words = [
    "vibrador anal",
    "consolador anal",
    "lubricante para",
    "gel lubricante",
]


def string_and_sex_shop(x: str) -> bool:
    is_sex_shop = False
    x = typical_string_processing(x)
    one_word_x = x.split(" ")
    for word in one_word_x:
        if word in sex_shop_one_word_words:
            is_sex_shop = True
            break
    two_word_x = x.split(" ")
    # Get the combination of the two words
    for i in range(len(two_word_x)):
        for j in range(i + 1, len(two_word_x)):
            if two_word_x[i] + " " + two_word_x[j] in sex_shop_two_word_words:
                is_sex_shop = True
                break
    return is_sex_shop


title_df_train_df["is_sex_shop"] = title_df_train_df["title"].apply(string_and_sex_shop)
title_df_train_df["is_sex_shop"].value_counts()
# %%
# check for used items in sex shop
title_df_train_df.loc[title_df_train_df["is_sex_shop"] & (title_df_train_df["y"] == 1)]
# %%
title_df_train_df.loc[title_df_train_df["is_sex_shop"] & (title_df_train_df["y"] == 0)]
# %%
title_df_train_df.loc[title_df_train_df["is_sex_shop"] == False][
    "title"
].value_counts()[140:160]

# %%
# Look for car titles
car_brands = [
    "chevrolet",
    "ford",
    "peugeot",
    "renault",
    "fiat",
    "toyota",
    "volkswagen",
    "honda",
    "hyundai",
    "nissan",
    "citroen",
    "citroën",
    "jeep",
    "auto",
    "camioneta",
    "subaru",
    "rover",
]

motorcycle_brands = [
    "honda",
    "yamaha",
    "suzuki",
    "harley-davidson",
    "kawasaki",
    "triumph",
    "bmw",
    "ducati",
    "motocicleta",
]


def string_and_car(x: str) -> bool:
    is_car = False
    x = typical_string_processing(x)
    for word in car_brands:
        if word in x:
            is_car = True
            break
    for word in motorcycle_brands:
        if word in x:
            is_car = True
            break
    return is_car


# %%
title_df_train_df["is_car"] = title_df_train_df["title"].apply(string_and_car)
title_df_train_df["is_car"].value_counts()
# %%
title_df_train_df.loc[title_df_train_df["is_car"] & (title_df_train_df["y"] == 1)]
# %%
title_df_train_df.loc[title_df_train_df["is_car"] & (title_df_train_df["y"] == 0)]
# %%
title_df_train_df.loc[title_df_train_df["is_car"] == False]["title"].value_counts()[
    120:140
]
# %%
title_df_train_df.groupby("is_car")["y"].value_counts()
# %%
# Clothes
clothes_words = [
    "campera",
    "camisa",
    "zapato",
    "vestido",
    "pantalon",
    "remera",
    "camiseta",
    "chaqueta",
    "bikini",
    "short",
    "zapatilla",
    "falda",
    "pollera",
    "zoquete",
    "soquete",
    "cartera",
    "caterita",
    "bolso",
    "mochila",
    "collar",
    "caravana",
    "pulsera",
    "chaleco",
    "traje",
    "corbata",
    "gorra",
    "gorro",
    "gafas",
    "botas",
    "tacones",
]


def string_and_clothes(x: str) -> bool:
    is_clothes = False
    x = typical_string_processing(x)
    for word in clothes_words:
        if word in x:
            is_clothes = True
            break
    return is_clothes


# %%
title_df_train_df["is_clothes"] = title_df_train_df["title"].apply(string_and_clothes)
title_df_train_df["is_clothes"].value_counts()
# %%
title_df_train_df.groupby("is_clothes")["y"].value_counts()


# %%
title_df_train_df.loc[title_df_train_df["is_clothes"] & (title_df_train_df["y"] == 1)]
# %%
title_df_train_df.loc[title_df_train_df["is_clothes"] & (title_df_train_df["y"] == 0)]
# %%
title_df_train_df.loc[
    ~title_df_train_df["is_clothes"]
    & ~title_df_train_df["is_car"]
    & ~title_df_train_df["is_sex_shop"]
]["title"].value_counts()[260:280]
# %%
# Home section
home_words = [
    "mesa",
    "silla",
    "tv",
    "parrilla",
    "velador",
    "cama",
    "sofa",
    "sillon",
    "escritorio",
    "cuadro",
    "almohadon",
    "almohada",
    "alfombra",
    "florero",
    "perchero",
    "comoda",
    "cortina",
    "espejo",
    "tazas",
    "platos",
    "vajilla",
    "cuchara",
    "cuchillo",
    "tenedor",
    "cazuela",
    "olla",
    "sarten",
    "tostadora",
    "exprimidor",
    "heladera",
    "microondas",
    "televisor",
    "lampara",
]


def string_and_home(x: str) -> bool:
    is_home = False
    x = typical_string_processing(x)
    for word in home_words:
        if word in x:
            is_home = True
            break
    return is_home


title_df_train_df["is_home"] = title_df_train_df["title"].apply(string_and_home)
title_df_train_df["is_home"].value_counts()
# %%
title_df_train_df.groupby("is_home")["y"].value_counts()
# %%
title_df_train_df.loc[title_df_train_df["is_home"] & (title_df_train_df["y"] == 1)]
# %%
title_df_train_df.loc[title_df_train_df["is_home"] & (title_df_train_df["y"] == 0)]
# %%
title_df_train_df.loc[
    ~title_df_train_df["is_clothes"]
    & ~title_df_train_df["is_car"]
    & ~title_df_train_df["is_sex_shop"]
    & ~title_df_train_df["is_home"]
]["title"].value_counts()[100:120]

# %%
electronics_words = [
    "laptop",
    "notebook",
    "televisor",
    "celular",
    "tablet",
    "teclado",
    "smartphone",
    "smartwatch",
    "iphone",
    "ipad",
    "samsung",
    "motorola",
    # "lg",
    "sony",
    "panasonic",
    "philips",
    # "jbl",
    "bose",
    "beats",
    "apple",
]


def string_and_electronics(x: str) -> bool:
    is_electronics = False
    x = typical_string_processing(x)
    for word in electronics_words:
        if word in x:
            is_electronics = True
            break
    return is_electronics


# %%
title_df_train_df["is_electronics"] = title_df_train_df["title"].apply(
    string_and_electronics
)
title_df_train_df["is_electronics"].value_counts()
# %%
title_df_train_df.loc[
    ~title_df_train_df["is_clothes"]
    & ~title_df_train_df["is_car"]
    & ~title_df_train_df["is_sex_shop"]
    & ~title_df_train_df["is_home"]
].groupby("is_electronics")["y"].value_counts()
# %%
title_df_train_df.loc[
    title_df_train_df["is_electronics"] & (title_df_train_df["y"] == 1)
].head(20)
# %%
title_df_train_df.loc[
    title_df_train_df["is_electronics"]
    & (title_df_train_df["y"] == 0)
    & ~title_df_train_df["is_clothes"]
    & ~title_df_train_df["is_car"]
    & ~title_df_train_df["is_sex_shop"]
    & ~title_df_train_df["is_home"]
].head(20)
# %%
title_df_train_df.loc[
    ~title_df_train_df["is_clothes"]
    & ~title_df_train_df["is_car"]
    & ~title_df_train_df["is_sex_shop"]
    & ~title_df_train_df["is_home"]
    & ~title_df_train_df["is_electronics"]
]["title"].value_counts()[25:50]


# %%
def is_nuevo_or_usado(x: str):
    x = typical_string_processing(x)
    if "nuevo" in x or "de fabrica" in x:
        if "casi nuevo" in x or "como nuevo" in x:
            return "usado"
        else:
            return "nuevo"
    elif "usado" in x or "segunda mano" in x:
        return "usado"
    else:
        return "unknown"


title_df_train_df["nuevo_or_usado"] = title_df_train_df["title"].apply(
    is_nuevo_or_usado
)
# %%
title_df_train_df.groupby("nuevo_or_usado")["y"].value_counts()
# %%
title_df_train_df.loc[title_df_train_df["nuevo_or_usado"] == "nuevo"][
    "title"
].value_counts()[25:50]
# %%
title_df_train_df.loc[title_df_train_df["nuevo_or_usado"] == "usado"][
    "title"
].value_counts()[0:25]
# %%
