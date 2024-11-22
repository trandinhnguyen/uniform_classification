import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from preprocessing import *
from models import *


class Evaluator:
    def __init__(
        self,
        model: L.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        saved_folder: str,
        save_pr_curve: bool,
    ):
        self.model = model
        self.dataloader = dataloader
        self.saved_folder = saved_folder
        self.save_pr_curve = save_pr_curve

        self.model.eval()

    def evaluate(self):
        # Only the probability of the positive class will be stored
        self.all_labels, self.all_probs, self.all_preds = [], [], []

        for images, labels in tqdm(self.dataloader):
            self.all_labels = np.concatenate((self.all_labels, labels.numpy()))
            images, labels = images.to(model.device), labels.to(model.device)

            with torch.no_grad():
                logits = model(images)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            # Only the probability of the positive class will be stored
            self.all_probs = np.concatenate((self.all_probs, probs[:, 1]))
            self.all_preds = np.concatenate((self.all_preds, preds))

        self.save_metrics()
        if self.save_pr_curve and (np.unique(self.all_labels).size == 2):
            self.save_pr_curve_plot()

    def save_metrics(self):
        report = metrics.classification_report(
            self.all_labels, self.all_preds, output_dict=True
        )
        output_filename = f"{self.saved_folder}/metric_scores.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

    def save_pr_curve_plot(self):
        """Only for binary classification"""
        eps = 1e-12
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            self.all_labels, self.all_probs
        )
        avg_precision = metrics.average_precision_score(self.all_labels, self.all_probs)
        f1_scores = 2 / (1 / (precisions + eps) + 1 / (recalls + eps))
        best_idx = f1_scores.argmax()

        plt.figure(figsize=(6, 6))
        plt.plot(recalls, precisions, label=f"AP={avg_precision:.3f}")
        plt.plot(
            recalls[best_idx],
            precisions[best_idx],
            "ro",
            label=f"Highest F1: T={thresholds[best_idx]:.3f}, "
            + f"P={precisions[best_idx]:.4f}, R={recalls[best_idx]:.4f}",
        )
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(f"{self.saved_folder}/prcurve.jpg")


if __name__ == "__main__":
    for ckpt_name in ["best", "last"]:
        ckpt_path = f"bidv_uniform_classification/j8vjy64f/checkpoints/{ckpt_name}.ckpt"

        splited_path = ckpt_path.split("/")
        saved_folder = os.path.join("output", splited_path[-3], splited_path[-1][:4])

        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)

        dt = BIDVUniformDataset("datasets/uniform_bidv/three_classes", 64)
        model = ViT.load_from_checkpoint(ckpt_path)

        evaluator = Evaluator(model, dt.dataloaders["test"], saved_folder, True)
        evaluator.evaluate()
