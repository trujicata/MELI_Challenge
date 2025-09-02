from challenge.new_or_used import build_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from lightning import LightningDataModule
import random
import pandas as pd
import numpy as np


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> tuple[str, torch.Tensor]:
        # Convert labels to float tensors for BCE loss
        label = torch.zeros(2)
        label[self.y[idx]] = 1
        return self.X[idx], label


class TitlesDataModule(LightningDataModule):
    def __init__(self, batch_size=128, num_workers=0) -> None:
        """
        Uses the build_dataset function to get the train and test data.
        Args:
            batch_size: int
            num_workers: int
        Returns:
            None
        """
        super().__init__()
        X_train, y_train, X_test, y_test = build_dataset()

        self.X_train_titles = pd.DataFrame(X_train)["title"]
        y_train = [y == "used" for y in y_train]
        self.y_train = np.array(y_train).astype(int)

        self.X_test_titles = pd.DataFrame(X_test)["title"]
        y_test = [y == "used" for y in y_test]
        self.y_test = np.array(y_test).astype(int)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Split the train into train and val 20 % randomly.
        Creates the train, val and test datasets.
        Args:
            stage: str
        Returns:
            None
        """
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

    def train_dataloader(self) -> DataLoader:
        """
        Returns the train dataloader.
        Args:
            None
        Returns:
            DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns the val dataloader.
        Args:
            None
        Returns:
            DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.
        Args:
            None
        Returns:
            DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
