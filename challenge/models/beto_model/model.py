from sentence_transformers import SentenceTransformer
from lightning import LightningModule
import torch
import matplotlib.pyplot as plt
import numpy as np


class TextClassification(LightningModule):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.model_backbone = SentenceTransformer("ITESM/sentece-embeddings-BETO")
        if freeze_backbone:
            for param in self.model_backbone.parameters():
                param.requires_grad = False
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.model_backbone.get_sentence_embedding_dimension(),
                self.model_backbone.get_sentence_embedding_dimension() / 2,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.model_backbone.get_sentence_embedding_dimension() / 2,
                self.model_backbone.get_sentence_embedding_dimension() / 4,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.model_backbone.get_sentence_embedding_dimension() / 4,
                2,
            ),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # Store device for later use
        self._device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Put everything to the same device
        self.mlp_head.to(self._device)
        self.loss_fn.to(self._device)

        self.val_conf_matrix = ConfusionMatrixPloter(classes=["new", "used"])
        self.train_conf_matrix = ConfusionMatrixPloter(classes=["new", "used"])
        self.test_conf_matrix = ConfusionMatrixPloter(classes=["new", "used"])

    def forward(self, x):
        x = self.model_backbone.encode(x, convert_to_tensor=True)
        # Convert inference tensor to regular tensor for gradient computation
        if hasattr(x, "detach"):
            x = x.detach().clone()
        # Move tensor to the same device as the model
        x = x.to(self._device)
        x = self.mlp_head(x)
        # Return raw logits for BCEWithLogitsLoss
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

        # Apply softmax for confusion matrix calculation
        probs = torch.softmax(logits, dim=1)
        class_predictions = probs.argmax(dim=1)
        preds = torch.zeros_like(probs)
        preds[torch.arange(probs.shape[0]), class_predictions] = 1
        self.train_conf_matrix.update(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

        # Apply softmax for confusion matrix calculation
        probs = torch.softmax(logits, dim=1)
        class_predictions = probs.argmax(dim=1)
        preds = torch.zeros_like(probs)
        preds[torch.arange(probs.shape[0]), class_predictions] = 1
        self.val_conf_matrix.update(preds, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)

        # Apply softmax for confusion matrix calculation
        probs = torch.softmax(logits, dim=1)
        class_predictions = probs.argmax(dim=1)
        preds = torch.zeros_like(probs)
        preds[torch.arange(probs.shape[0]), class_predictions] = 1
        self.test_conf_matrix.update(preds, y)

        return loss

    def on_fit_start(self):
        """Called when fit begins"""
        # Store the device where the model is actually placed
        self._device = next(self.parameters()).device

    def on_test_epoch_end(self):
        precision, recall, f1, accuracy = self.calculate_metrics(
            self.test_conf_matrix.compute()
        )
        self.log_conf_matrix(mode="test")
        self.log_dict(
            {
                # "test_precision": precision,
                # "test_recall": recall,
                # "test_f1": f1,
                "test_accuracy": accuracy,
            }
        )

    def on_validation_epoch_end(self) -> None:
        precision, recall, f1, accuracy = self.calculate_metrics(
            self.val_conf_matrix.compute()
        )
        self.log_conf_matrix(mode="val")
        self.log_dict(
            {
                # "val_precision": precision,
                # "val_recall": recall,
                # "val_f1": f1,
                "val_accuracy": accuracy,
            }
        )

    def on_train_epoch_end(self) -> None:
        precision, recall, f1, accuracy = self.calculate_metrics(
            self.train_conf_matrix.compute()
        )
        self.log_conf_matrix(mode="train")
        self.log_dict(
            {
                # "train_precision": precision,
                # "train_recall": recall,
                # "train_f1": f1,
                "train_accuracy": accuracy,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mlp_head.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.01,
        )
        return [optimizer], [scheduler]

    def log_conf_matrix(self, mode="val"):
        if mode == "val":
            fig = self.val_conf_matrix.plot()
            name = "Validation_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.val_conf_matrix.reset()

        elif mode == "test":
            fig = self.test_conf_matrix.plot()
            name = "Test_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.test_conf_matrix.reset()

        else:
            fig = self.train_conf_matrix.plot()
            name = "Train_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.train_conf_matrix.reset()
        plt.close()

    def calculate_metrics(self, cm: np.array):
        """
        Calculate the precision, recall and f1 score from the confusion matrix.
        Args:
            cm: np.array
        Returns:
            precision: float
            recall: float
            f1: float
            accuracy: float
        """
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = np.diag(cm) / np.sum(cm)
        for metric in [precision, recall, f1, accuracy]:
            metric[np.isnan(metric)] = 0
            metric = torch.Tensor(metric)

        return precision.mean(), recall.mean(), f1.mean(), accuracy.mean()


class ConfusionMatrixPloter:
    def __init__(self, classes: list[str]):
        """
        Initialize the confusion matrix plotter.
        Args:
            classes: list[str]
        Returns:
            None
        """
        self.num_classes = len(classes)
        self.classes = classes
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, targets):
        """
        Update the confusion matrix.
        Args:
            preds: torch.Tensor
            targets: torch.Tensor
        Returns:
            None
        """
        conf_matrix = self.confusion_matrix(
            preds.detach().cpu(), targets.detach().cpu()
        ).numpy()
        self.matrix += conf_matrix

    def compute(self):
        """
        Return the confusion matrix.
        Args:
            None
        Returns:
            np.array
        """
        return self.matrix

    def plot(self):
        """
        Plot the confusion matrix.
        Args:
            None
        Returns:
            plt.figure
        """
        plt.figure(figsize=(8, 8))
        normalized_matrix = self.matrix / self.matrix.sum(axis=1, keepdims=True)

        plt.imshow(normalized_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(
                    j,
                    i,
                    round(normalized_matrix[i, j], 2),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=26,
                )
        return plt.gcf()

    def reset(self):
        """
        Reset the confusion matrix.
        Args:
            None
        Returns:
            None
        """
        self.matrix *= 0

    def confusion_matrix(self, preds, target):
        matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int)
        for p, t in zip(preds, target):
            pred_class = torch.argmax(p)
            target_class = torch.argmax(t)
            matrix[target_class][pred_class] += 1
        return matrix
