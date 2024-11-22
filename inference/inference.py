import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        vit.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(768, 2))
        self.vit = vit

    def forward(self, inputs):
        return self.vit(inputs).logits


def preprocess(img) -> torch.Tensor:
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return torch.tensor(img)


def init_model(checkpoint: str):
    state_dict = torch.load(checkpoint, weights_only=True)["state_dict"]
    classifier = ViT()
    classifier.load_state_dict(state_dict)
    classifier.eval()
    classifier.to("cuda")
    return classifier


def classify(classifier, img: np.ndarray):
    img = preprocess(img)
    with torch.no_grad():
        output = classifier(img.to("cuda"))
    return output.argmax(1)[0].item()


if __name__ == "__main__":
    classifier = init_model("uniform_classification/easop82o/checkpoints/best.ckpt")
    img = cv2.imread("photo_2024-11-14_17-59-49.jpg")
    output = classify(classifier, img)
    print(output)
    img = cv2.imread("photo_2024-11-14_17-59-50.jpg")
    output = classify(classifier, img)
    print(output)
