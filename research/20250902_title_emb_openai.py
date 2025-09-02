# %%
import start  # noqa
from challenge.new_or_used import build_dataset
import pandas as pd
import numpy as np

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
import os
from openai import OpenAI


client = OpenAI(api_key="...")
response = client.embeddings.create(
    input=train_titles.iloc[0], model="text-embedding-3-small"
)

print(response.data[0].embedding)

# %%
sentences = train_titles.to_list()  # list of strings
sentences

# %%
from tqdm import tqdm

embeddings = []
# Pair the sentence with the y_train label
sentences_with_labels = list(zip(sentences, y_train))

for sentence, label in tqdm(sentences_with_labels):
    try:
        response = client.embeddings.create(
            input=sentence, model="text-embedding-3-small"
        )
        embd = np.array(response.data[0].embedding)
        embeddings.append((embd, label))
    except Exception as e:
        print(f"Error embedding sentence: {sentence}")
        print(e)
        continue

print(f"Number of embeddings: {len(embeddings)}")
print(f"First embedding shape: {embeddings[0].shape}")
print(
    f"All embeddings have same shape: {all(emb.shape == embeddings[0].shape for emb in embeddings)}"
)
# %%
# Save all the embeddings to a pickle file
import pickle

with open("embeddings_openai.pkl", "wb") as f:
    pickle.dump(embeddings, f)
# %%

# %%
# TensorBoard logging for embeddings visualization
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm

# Create a unique log directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"tensorboard_logs/embeddings_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Convert list of numpy arrays to a single torch tensor
# Stack all embeddings into a single tensor
embeddings_array = np.stack(embeddings)  # Shape: (num_embeddings, embedding_dim)
embeddings_tensor = torch.from_numpy(embeddings_array).float()
print(f"Stacked embeddings shape: {embeddings_tensor.shape}")

# Get corresponding labels for the training data
# Since we're using train_titles, we need the corresponding y_train labels
# We'll use the first len(sentences) labels from y_train
labels = y_train[: len(sentences)]
# %%
# Create metadata for each embedding 1 if label=="used" and 0 if label=="new"
metadata = []
for label in tqdm(labels):
    metadata.append(label)


# %%
# Log embeddings to TensorBoard
writer.add_embedding(
    embeddings_tensor,
    metadata=metadata,
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

#
j
