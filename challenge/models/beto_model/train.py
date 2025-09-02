from challenge.models.beto_model.model import TextClassification
from challenge.models.beto_model.dataset import TitlesDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    model = TextClassification(freeze_backbone=True)
    data_module = TitlesDataModule()
    logger = TensorBoardLogger(
        "lightning_logs",
        name="beto_model_title_classification",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="lightning_logs/checkpoints",
        filename=f"checkpoint-{{epoch:02d}}-{{val_loss:.2f}}-{{val_accuracy:.2f}}",
        save_top_k=3,
        monitor="val_accuracy",
        mode="max",
    )

    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        accelerator="mps",
        log_every_n_steps=5,
        gradient_clip_val=0.5,
        logger=logger,
    )
    trainer.fit(model, data_module)

    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
