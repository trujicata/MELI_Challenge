# %%
import random

import numpy as np
import pandas as pd
import start  # noqa
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

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

# %%
from sentence_transformers import SentenceTransformer

sentences = train_titles[:2]

model = SentenceTransformer("ITESM/sentece-embeddings-BETO")
embeddings = model.encode(sentences)
print(embeddings)

# %%


# %%
class TextClassification(LightningModule):
    def __init__(self, model_backbone, freeze_backbone: bool = True):
        super().__init__()
        self.model_backbone = model_backbone
        if freeze_backbone:
            for param in self.model_backbone.parameters():
                param.requires_grad = False
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(model_backbone.get_sentence_embedding_dimension(), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="binary")
        # Store device for later use
        self._device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Put everything to the same device
        self.mlp_head.to(self._device)
        self.loss_fn.to(self._device)
        self.accuracy.to(self._device)

    def forward(self, x):
        x = self.model_backbone.encode(x, convert_to_tensor=True)
        # Convert inference tensor to regular tensor for gradient computation
        if hasattr(x, "detach"):
            x = x.detach().clone()
        # Move tensor to the same device as the model
        x = x.to(self._device)
        x = self.mlp_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.accuracy(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy(y_hat, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", self.accuracy(y_hat, y))
        return loss

    def on_fit_start(self):
        """Called when fit begins"""
        # Store the device where the model is actually placed
        self._device = next(self.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %%


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert labels to float tensors for BCE loss
        label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        return self.X[idx], label


# %%


class TitlesDataModule(LightningDataModule):
    def __init__(
        self, train_titles, test_titles, train_y, test_y, batch_size=32, num_workers=4
    ):
        super().__init__()
        self.X_train_titles = train_titles
        self.X_test_titles = test_titles
        self.y_train = train_y
        self.y_test = test_y
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Split train into train and val 20 % randomly
        random.seed(42)
        val_size = int(len(self.X_train_titles) * 0.2)

        self.X_train_and_val_titles = self.X_train_titles
        self.y_train_and_val = self.y_train

        random_indices = random.sample(
            range(len(self.X_train_and_val_titles)), val_size
        )
        self.X_train_titles = [
            self.X_train_and_val_titles[i]
            for i in range(len(self.X_train_and_val_titles))
            if i not in random_indices
        ]
        self.y_train = [
            self.y_train_and_val[i]
            for i in range(len(self.y_train_and_val))
            if i not in random_indices
        ]
        self.X_val_titles = [self.X_train_and_val_titles[i] for i in random_indices]
        self.y_val = [self.y_train_and_val[i] for i in random_indices]

        self.train_dataset = TextDataset(self.X_train_titles, self.y_train)
        self.val_dataset = TextDataset(self.X_val_titles, self.y_val)
        self.test_dataset = TextDataset(self.X_test_titles, self.y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# %%
data_module = TitlesDataModule(
    train_titles, test_titles, y_train, y_test, num_workers=0
)
data_module.setup()
# %%
data_module.train_dataloader()

# %%
text_classification_model = TextClassification(model_backbone=model)

# %%
# Test the model with a single title
with torch.no_grad():
    test_output = text_classification_model([train_titles[0]])
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output: {test_output}")
# %%
model_name = "text_classification_model"
checkpoint_callback = ModelCheckpoint(
    dirpath=f"lightning_logs/checkpoints/{model_name}",
    filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_accuracy:.2f}}",
    monitor="val_accuracy",
    mode="max",
    save_top_k=3,
)
trainer = Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback],
    accelerator="mps",
    log_every_n_steps=5,
    gradient_clip_val=0.5,
)
# %%
trainer.fit(
    text_classification_model,
    datamodule=data_module,
)
# %%
trainer.test(text_classification_model, datamodule=data_module, ckpt_path="best")

# Multiprocessing guard for macOS compatibility
if __name__ == "__main__":
    # Your main execution code can go here if needed
    pass
