import torch
import torchvision
import lightning as L
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    accuracy,
    binary_accuracy,
    multiclass_accuracy,
)
from transformers import ViTForImageClassification


class Model(L.LightningModule):
    def __init__(self, lr, class_weights):
        super().__init__()
        self.lr = lr
        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.class_weights = torch.tensor(class_weights)

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        result_dict = self._get_preds_loss_metrics(batch, "train")
        self.log_dict(
            result_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )
        return result_dict["train/loss"]

    def validation_step(self, batch, batch_idx):
        result_dict = self._get_preds_loss_metrics(batch, "val")
        self.log_dict(
            result_dict,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [lr_scheduler_config]

    def _get_preds_loss_metrics(self, batch, phase):
        """Convenient function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        preds = logits.argmax(1)

        # must set device location of class weights here to automatically
        # choose device if using multiple gpus
        loss = F.cross_entropy(logits, y, weight=self.class_weights.cuda())
        acc = torchmetrics.functional.accuracy(preds, y, "binary")
        return {f"{phase}/loss": loss, f"{phase}/acc": acc}


class ViT(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(lr, class_weights)
        vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        vit.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(768, n_classes))
        self.vit = vit

    def forward(self, inputs):
        return self.vit(inputs).logits


class MobileNetV2(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(lr, class_weights)
        model = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        )
        model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(1280, n_classes))
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = MobileNetV2(1)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    print(output)
