import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

from preprocessing import *
from models import *


def train(
    model,
    dataloaders,
    precision,
    n_epochs,
    project,
    name,
    find_lr,
    ckpt_path=None,
):
    # Define callbacks
    ckpt_cb = ModelCheckpoint(
        filename="best",
        monitor="val/acc",
        save_last=True,
        save_top_k=1,
        mode="max",
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping_cb = EarlyStopping(monitor="val/loss", patience=10, mode="min")

    # Define logger
    wandb_logger = WandbLogger(name=name, project=project)

    # Create trainer
    trainer = L.Trainer(
        # accelerator="cpu",
        precision=precision,
        logger=wandb_logger,
        callbacks=[
            # early_stopping_cb,
            ckpt_cb,
            lr_monitor,
        ],
        max_epochs=n_epochs,
        num_sanity_val_steps=2,
    )

    if find_lr:
        tuner = Tuner(trainer)
        # Run learning rate finder
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
        # Results can be found in
        # print(lr_finder.results)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_plot.jpg")
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        # update hparams of the model
        model.hparams.lr = new_lr

    # Fitting
    trainer.fit(
        model=model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=ckpt_path,  # Resume training state
    )


if __name__ == "__main__":
    batch_size = 32
    n_epochs = 70
    project = "bidv_uniform_classification"
    name = "vit"

    dt = BIDVUniformDataset("datasets/uniform_bidv/three_classes")
    model = ViT(n_classes=3, lr=1e-4, class_weights=[0.451774, 0.182833, 0.365400])
    train(
        model,
        dt.dataloaders,
        precision="bf16-mixed",
        n_epochs=n_epochs,
        project=project,
        name=name,
        find_lr=False,
    )
