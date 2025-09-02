# %%
import start  # noqa
from challenge.new_or_used import build_dataset
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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
inputs = tokenizer(
    train_titles.to_list(), return_tensors="pt", truncation=True, padding=True
).to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
# %%
attention_mask = inputs["attention_mask"]
embeddings = outputs.last_hidden_state
mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
sentence_embeddings = torch.sum(embeddings * mask_expanded, 1) / mask_expanded.sum(1)
# %%
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm

# Create a unique log directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"tensorboard_logs/embeddings_roberta_{timestamp}"
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
