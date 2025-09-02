import pandas as pd
from challenge.dataset.categories import preprocess_categories
from challenge.dataset.variations import preprocess_variations
from challenge.dataset.non_mercadopago_payments import (
    preprocess_non_mercadopago_payments,
)
from challenge.dataset.warranty import preprocess_warranty
from challenge.dataset.seller import preprocess_seller_address, preprocess_seller_id
from challenge.dataset.shipping import preprocess_shipping
from challenge.dataset.prices import preprocess_prices
from challenge.dataset.pictures import preprocess_pictures
from challenge.dataset.quantities import preprocess_quantities


DROPPING_COLUMNS = [
    "site_id",
    "listing_source",
    "coverage_areas",
    "international_delivery_mode",
    "differential_pricing",
    "subtitle",
    "sub_status",
    "deal_ids",
    "attributes",
    "id",
    "official_store_id",
    "thumbnail",
    "accepts_mercadopago",
    "secure_thumbnail",
    "descriptions",
    "stop_time",
    "video_id",
    "catalog_product_id",
    "permalink",
    "title",
    "tags",
    "parent_item_id",
    "date_created",
    "status",
    "start_time",
]

FEATURES = [
    "seller_address_state",
    "most_used_seller_address_cities",
    "most_new_seller_address_cities",
    "seller_id",
    "variations",
    "tarjeta",
    "transferencia_bancaria",
    "acordar_comprador",
    "num_payment_methods_non_mercadopago",
    "efectivo",
    "warranty",
    "shipping_local_pick_up",
    "shipping_free_shipping",
    "shipping_method",
    "shipping_mode",
    "listing_type_id",
    "price_usd",
    "buying_mode",
    "has_parent_item_id",
    "popular_category",
    "pictures_max_size",
    "automatic_relist",
    "initial_quantity",
    "sold_quantity",
    "available_quantity",
]


def preprocess_whole_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing seller info...")
    df = preprocess_seller_address(df)
    df = preprocess_seller_id(df)
    print("Preprocessing variations...")
    df = preprocess_variations(df)
    print("Preprocessing non mercado pago payments...")
    df = preprocess_non_mercadopago_payments(df)
    print("Preprocessing warranty...")
    df = preprocess_warranty(df)
    print("Preprocessing shipping...")
    df = preprocess_shipping(df)
    print("Preprocessing prices...")
    df = preprocess_prices(df)
    print("Preprocessing categories...")
    df = preprocess_categories(df)
    print("Preprocessing has parent item id...")
    df["has_parent_item_id"] = df["parent_item_id"].notna()
    print("Preprocessing pictures...")
    df = preprocess_pictures(df)
    print("Preprocessing quantities...")
    df = preprocess_quantities(df, "initial_quantity")
    df = preprocess_quantities(df, "sold_quantity", possible_zeros=True)
    df = preprocess_quantities(df, "available_quantity", possible_zeros=True)
    df.drop(columns=DROPPING_COLUMNS, inplace=True)
    df = df[FEATURES]
    df.dropna(inplace=True)
    return df
