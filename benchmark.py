import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import *
from models import *


def get_results(logits, labels, all_labels, all_probs, all_preds):
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
    all_probs = np.concatenate((all_probs, probs[:, 1]))
    all_preds = np.concatenate((all_preds, preds))
    return all_labels, all_probs, all_preds


def save_metrics(labels, preds, folder, suffix):
    report = metrics.classification_report(
        labels, preds, target_names=["no", "yes"], output_dict=True
    )
    output_filename = f"output/{folder}/metric_scores_{suffix}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)


def save_pr_curve_plot(all_labels, all_probs, folder, suffix):
    eps = 1e-12
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        all_labels, all_probs
    )
    avg_precision = metrics.average_precision_score(all_labels, all_probs)
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
    plt.savefig(f"output/{folder}/prcurve_{suffix}.jpg")


def evaluate_single_dataset(
    dataloader,
    model,
    folder,
    file_suffix,
    save_pr_curve=False,
):
    # Only the probability of the positive class will be stored
    all_labels, all_probs, all_preds = [], [], []
    model.eval()

    for images, labels in tqdm(dataloader):
        images, labels = images.to(model.device), labels.to(model.device)
        with torch.no_grad():
            logits = model(images)

        all_labels, all_probs, all_preds = get_results(
            logits, labels, all_labels, all_probs, all_preds
        )

    save_metrics(all_labels, all_preds, folder, file_suffix)
    if save_pr_curve:
        save_pr_curve_plot(all_labels, all_probs, folder, file_suffix)


if __name__ == "__main__":
    for ckpt_name in ["best", "last"]:
        ckpt_path = f"uniform_classification/easop82o/checkpoints/{ckpt_name}.ckpt"

        split_path = ckpt_path.split("/")
        saved_folder = os.path.join(split_path[-3], split_path[-1][:4])
        if not os.path.exists(f"output/{saved_folder}"):
            os.makedirs(f"output/{saved_folder}")

        image_dataloaders = uniform_dataloaders("datasets/uniform", 32)
        model = ViT.load_from_checkpoint(ckpt_path)
        evaluate_single_dataset(
            image_dataloaders["val"],
            model,
            saved_folder,
            "uniform",
            save_pr_curve=True,
        )
