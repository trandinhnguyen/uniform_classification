import os

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
import timm

import onnx
from onnxconverter_common import float16

# from fastervit import create_model


class Model(L.LightningModule):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__()
        self.lr = lr
        self.example_input_array = torch.randn((1, 3, 224, 224))

        assert len(class_weights) == n_classes
        self.n_classes = n_classes
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
        acc = torchmetrics.functional.accuracy(
            preds, y, "multiclass", num_classes=self.n_classes, average="micro"
        )
        return {f"{phase}/loss": loss, f"{phase}/acc": acc}

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def to_onnx_fp16(self, output_path, **kwargs):
        self.to_onnx(output_path, **kwargs)
        onnx_fp32 = onnx.load_model(output_path)
        onnx_fp16 = float16.convert_float_to_float16(onnx_fp32, keep_io_types=True)

        root, ext = os.path.splitext(output_path)
        onnx_fp16_path = f"{root}_fp16{ext}"
        onnx.save(onnx_fp16, onnx_fp16_path)
        return onnx_fp16_path


class ViTBase(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(n_classes, lr, class_weights)
        vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        vit.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(768, n_classes))
        self.vit = vit

    def forward(self, inputs):
        return self.vit(inputs).logits


class MobileNetV2(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(n_classes, lr, class_weights)
        model = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        )
        model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(1280, n_classes))
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


class MobileNetV3(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(n_classes, lr, class_weights)
        self.model = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        )
        self.model.classifier[-1] = nn.Linear(1024, n_classes)

    def forward(self, inputs):
        return self.model(inputs)


class ViTTiny(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(n_classes, lr, class_weights)
        self.model = timm.create_model(
            "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            num_classes=n_classes,
        )

    def forward(self, inputs):
        return self.model(inputs)


# TODO: BUG
class FasterVit(Model):

    def __init__(self, n_classes, lr, class_weights):
        super().__init__(n_classes, lr, class_weights)
        self.model = create_model(
            "faster_vit_0_224", pretrained=True, model_path="faster_vit_0.pth.tar"
        )

    def forward(self, inputs):
        return self.model(inputs)


class FastViT(Model):
    def __init__(self, n_classes, lr, class_weights):
        super().__init__(n_classes, lr, class_weights)
        self.model = timm.create_model(
            "fastvit_t8.apple_in1k", pretrained=True, num_classes=n_classes
        )

    def forward(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = MobileNetV3(3, 1, [1] * 3)

    model.eval()
    print(model)

    with torch.no_grad():
        out = model(torch.randn(10, 3, 224, 224))

    print(out.shape)
    print(f"{model.num_params():,}")
