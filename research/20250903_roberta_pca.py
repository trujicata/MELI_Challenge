# %%
import numpy as np
import pandas as pd
import start  # noqa
import torch
from transformers import AutoModel, AutoTokenizer

from challenge.new_or_used import build_dataset

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
test_titles.value_counts(normalize=True)
# %%
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
# %%
model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base", dtype=torch.float16)
# %%
inputs = tokenizer(train_titles[0], return_tensors="pt", truncation=True).to(
    model.device
)
# %%
sample_sentences = train_titles[:10].to_list()

inputs = tokenizer(
    sample_sentences, return_tensors="pt", truncation=True, padding=True
).to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# %%
attention_mask = inputs["attention_mask"]
embeddings = outputs.last_hidden_state
mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
sentence_embeddings = torch.sum(embeddings * mask_expanded, 1) / mask_expanded.sum(1)
# %%
# Now for the whole dataset
# Use batches of 32
import tqdm

batch_size = 32
num_batches = len(train_titles) // batch_size + 1
embeddings_list = []
train_titles_list = train_titles.to_list()
for i in tqdm.tqdm(range(num_batches)):
    batch = train_titles_list[i * batch_size : (i + 1) * batch_size]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    sentence_embeddings = torch.sum(embeddings * mask_expanded, 1) / mask_expanded.sum(
        1
    )
    embeddings_list.append(sentence_embeddings)

# %%
embeddings_tensor = torch.cat(embeddings_list)
# %%
print(f"Stacked embeddings shape: {embeddings_tensor.shape}")
# %%
embeddings_array = embeddings_tensor.cpu().numpy()
from sklearn.cluster import KMeans
# Find the PCA of the embeddings, with variance explained of 0.8
# %%
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import normalize

# %%
pca = PCA(n_components=0.8)
pca.fit(embeddings_array)
# %%
print(f"Number of components: {pca.n_components_}")
# %%
embeddings_array_pca = pca.transform(embeddings_tensor)

# %%
X = embeddings_array_pca

# (Optional but recommended for text embeddings)
# L2-normalize to make Euclidean distance behave more like cosine distance for KMeans
Xn = normalize(X)


# ---- Search best K with multiple metrics ----
def pick_best_k(
    X,
    k_values=range(5, 21),
    n_init=20,
    max_iter=300,
    random_state=42,
):
    results = []

    X_sil = X

    best_k = None
    best_sil = -1.0
    best_model = None

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        ).fit(X)

        inertia = km.inertia_

        # Metrics
        # (Silhouette uses sample for speed if large; others on full set)
        sil = silhouette_score(X_sil, km.predict(X_sil))

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
            }
        )

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = km

    return best_k, best_model, results


# %%

# Try a sensible range; adjust upper bound depending on your dataset size
k_range = range(5, 21)
best_k, best_kmeans, metrics_table = pick_best_k(Xn, k_values=k_range)

print("KMeans model selection (higher Silhouette/CH is better; lower DB is better):")
for row in metrics_table:
    print(
        f"k={row['k']:>2} | inertia={row['inertia']:.2f} | "
        f"sil={row['silhouette']:.4f} | CH={row['calinski_harabasz']:.1f} | DB={row['davies_bouldin']:.4f}"
    )
print(f"\nSelected k by Silhouette: {best_k}")
# %%
# Final labels from the best model
best_kmeans.labels_
# %%
best_kmeans.cluster_centers_
import os
from datetime import datetime

# %%
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Create a unique log directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"tensorboard_logs/embeddings_roberta_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Convert list of numpy arrays to a single torch tensor
# Stack all embeddings into a single tensor
print(f"Stacked embeddings shape: {embeddings_tensor.shape}")

# Get corresponding labels for the training data
# Since we're using train_titles, we need the corresponding y_train labels
# We'll use the first len(sentences) labels from y_train
labels = y_train[: len(embeddings_tensor)]
# %%
# Create metadata for each embedding 1 if label=="used" and 0 if label=="new"
# Also add the cluster label
metadata = [
    [str(labels[i]), str(kmeans.labels_[i])] for i in range(len(embeddings_tensor))
]

# %%
# Log embeddings to TensorBoard
writer.add_embedding(
    embeddings_tensor,  # shape [N, D], CPU tensor
    metadata=metadata,  # list of [label, cluster]
    metadata_header=["label", "cluster"],
    global_step=0,
    tag="title_embeddings",
)

# Close the writer
writer.close()

print(f"Embeddings logged to TensorBoard in directory: {log_dir}")
print("To view the embeddings, run:")
print(f"tensorboard --logdir={log_dir}")
print("Then open http://localhost:6006 in your browser")

# %%
# %%
# Show all the titles belonging to cluster 0
df = df[: len(embeddings_tensor)]
df["kmeans_title_label"] = best_kmeans.labels_
df.head()
# %%
df.loc[df["kmeans_title_label"] == 1]["title"].value_counts()[:25]
# %%
df["used"] = y_train[: len(embeddings_tensor)]
df["used"] = df["used"].astype(int)
df["used"].value_counts()


# %%
df.groupby("kmeans_title_label")["used"].mean()

# %%
df.groupby("kmeans_title_label")["used"].count()
# %%
# Save embeddings
import pickle

with open("embeddings_roberta.pkl", "wb") as f:
    pickle.dump(embeddings_tensor.cpu().numpy(), f)
# %%
with open("embeddings_roberta.pkl", "rb") as f:
    embeddings_tensor = pickle.load(f)
# %%
embeddings_tensor.shape
# %%

# Save best kmeans
with open("best_kmeans.pkl", "wb") as f:
    pickle.dump(best_kmeans, f)
# %%
with open("best_kmeans.pkl", "rb") as f:
    best_kmeans = pickle.load(f)
# %%
# save pca
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)
# %%
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)
# %%
pca.n_components_
# %%
random_vector_798_dimensions = np.random.rand(768)
random_vector_798_dimensions.shape
# %%
transformed_vector_90_dimensions = pca.transform(
    random_vector_798_dimensions.reshape(1, -1)
)
# %%
transformed_vector_90_dimensions.shape

# %%
# define cluster

cluster_label = best_kmeans.predict(transformed_vector_90_dimensions)
# %%
cluster_label == 4
# %%
