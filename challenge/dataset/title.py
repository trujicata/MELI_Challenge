from challenge.dataset.utils import typical_string_processing
import pandas as pd
from tqdm import tqdm
import torch
import pickle
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize


SEX_SHOP_WORDS = {
    "ONE-WORDS": [
        "consolador",
        "vibrador",
        "anal",
        "pene",
        "peneano",
        "masajes",
    ],
    "TWO-WORDS": [
        "vibrador anal",
        "consolador anal",
        "lubricante para",
        "gel lubricante",
    ],
}

CAR_RELATED_WORDS = [
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

CLOTHES_WORDS = [
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

HOME_WORDS = [
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

ELECTRONICS_WORDS = [
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
    "sony",
    "panasonic",
    "philips",
    "bose",
    "beats",
    "apple",
]


def string_and_sex_shop(x: str) -> bool:
    is_sex_shop = False
    x = typical_string_processing(x)
    one_word_x = x.split(" ")
    for word in one_word_x:
        if word in SEX_SHOP_WORDS["ONE-WORDS"]:
            is_sex_shop = True
            break
    two_word_x = x.split(" ")
    # Get the combination of the two words
    for i in range(len(two_word_x)):
        for j in range(i + 1, len(two_word_x)):
            if two_word_x[i] + " " + two_word_x[j] in SEX_SHOP_WORDS["TWO-WORDS"]:
                is_sex_shop = True
                break
    return is_sex_shop


def car_appearance(x: str) -> bool:
    for word in CAR_RELATED_WORDS:
        if word in x:
            return True
    return False


def clothes_appearance(x: str) -> bool:
    for word in CLOTHES_WORDS:
        if word in x:
            return True
    return False


def home_appearance(x: str) -> bool:
    for word in HOME_WORDS:
        if word in x:
            return True
    return False


def electronics_appearance(x: str) -> bool:
    for word in ELECTRONICS_WORDS:
        if word in x:
            return True
    return False


def is_nuevo_or_usado(x: str) -> str:
    if "nuevo" in x or "de fabrica" in x:
        if "casi nuevo" in x or "como nuevo" in x:
            return "usado"
        else:
            return "nuevo"
    elif "usado" in x or "segunda mano" in x:
        return "usado"
    else:
        return "unknown"


def cluster_title(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """
    Transform the title into an embedding, then transform it into 90
    dimensions using PCA, then cluster the embeddings using KMeans.
    Args:
        df (pd.DataFrame): The dataframe to process
        batch_size (int): The batch size for the embedding
    Returns:
        pd.DataFrame: The processed dataframe

    Columns added:
        kmeans_title_label_0: bool
        kmeans_title_label_1: bool
        kmeans_title_label_2: bool
        kmeans_title_label_3: bool
        kmeans_title_label_4: bool
    """

    titles_list = df["title"].to_list()
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModel.from_pretrained(
        "FacebookAI/xlm-roberta-base", dtype=torch.float16
    )
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    num_batches = len(titles_list) // batch_size + 1
    embeddings_list = []
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(titles_list))
        batch = titles_list[start_idx:end_idx]
        if not batch:
            break
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sentence_embeddings = torch.sum(
            embeddings * mask_expanded, 1
        ) / mask_expanded.sum(1)
        embeddings_list.append(sentence_embeddings)
    embeddings_tensor = torch.cat(embeddings_list)

    pca = pickle.load(open("../eda/PCA_roberta.pkl", "rb"))
    embeddings_tensor_pca = pca.transform(embeddings_tensor.cpu().numpy())
    embeddings_normalized = normalize(embeddings_tensor_pca)
    kmeans = pickle.load(open("../eda/best_kmeans_roberta.pkl", "rb"))
    cluster_labels = kmeans.predict(embeddings_normalized)
    # One hot encode the cluster labels
    for k in range(kmeans.n_clusters):
        df[f"kmeans_title_label_{k}"] = cluster_labels == k
    return df


def title_string_processing(
    df: pd.DataFrame, use_encoders: bool = False
) -> pd.DataFrame:
    """
    Process the title column of the dataframe
    Args:
        df (pd.DataFrame): The dataframe to process
    Returns:
        pd.DataFrame: The processed dataframe

    Columns added:
        title_sex_shop: bool
        title_car: bool
        title_clothes: bool
        title_home: bool
        title_electronics: bool
        title_nuevo_or_usado: str

    Columns dropped:
        title

    """
    df["title"] = df["title"].apply(typical_string_processing)
    df["title_sex_shop"] = df["title"].apply(string_and_sex_shop)
    df["title_car"] = df["title"].apply(car_appearance)
    df["title_clothes"] = df["title"].apply(clothes_appearance)
    df["title_home"] = df["title"].apply(home_appearance)
    df["title_electronics"] = df["title"].apply(electronics_appearance)
    df["title_nuevo_or_usado"] = df["title"].apply(is_nuevo_or_usado)
    if use_encoders:
        df = cluster_title(df)
    df.drop(columns=["title"], inplace=True)
    return df
