import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional
from transformers import ViTImageProcessor


def pad_to_aspect_ratio(image, aspect_ratio=0.75):
    w, h = image.size
    target_w, target_h = w, h

    if w / h < aspect_ratio:  # Image is too tall, pad width
        target_w = int(h * aspect_ratio)
    elif w / h > aspect_ratio:  # Image is too wide, pad height
        target_h = int(w / aspect_ratio)

    pad_w = (target_w - w) // 2
    pad_h = (target_h - h) // 2

    # Pad in (left, top, right, bottom) order
    padding = (pad_w, pad_h, target_w - w - pad_w, target_h - h - pad_h)
    return T.functional.pad(image, padding, fill=255)  # Fill with white


class ImageDatasetFromCSV(torch.utils.data.Dataset):
    def __init__(self, path: str, transforms: T.Compose):
        self.dataset_df = pd.read_csv(path)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df.index)

    def __getitem__(self, index):
        item = self.dataset_df.iloc[index]
        path = item["path"]
        image = Image.open(path)
        image = self.transforms(image)
        label = item["label"]
        return image, label


class BIDVUniformDataset:

    def __init__(self, root, batch_size=32, num_workers=8):
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.image_mean = processor.image_mean
        self.image_std = processor.image_std
        size = processor.size["height"]

        train_transforms = T.Compose(
            [
                T.Lambda(lambda img: pad_to_aspect_ratio(img, aspect_ratio=0.75)),
                T.RandomPerspective(fill=255),
                T.RandomRotation((0, 360), fill=255),
                T.Resize(size),
                T.RandomCrop(size),
                T.ToTensor(),
                T.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        val_transforms = T.Compose(
            [
                T.Lambda(lambda img: pad_to_aspect_ratio(img, aspect_ratio=0.75)),
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

        data_transforms = {
            "train": train_transforms,
            "val": val_transforms,
            "test": val_transforms,
        }

        self.datasets = {
            x: ImageDatasetFromCSV(f"{root}/{x}.csv", data_transforms[x])
            for x in ["train", "val", "test"]
        }

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=batch_size,
                shuffle=(x == "train"),
                num_workers=num_workers,
            )
            for x in ["train", "val", "test"]
        }

    def visualize(self, subset="train"):
        # get a batch from dataloader
        inputs, classes = next(iter(self.dataloaders[subset]))

        # make a grid from batch
        grid = torchvision.utils.make_grid(inputs)

        mean = np.array(self.image_mean).reshape(1, 1, 3)
        std = np.array(self.image_std).reshape(1, 1, 3)

        # imshow for tensor
        grid = grid.numpy().transpose((1, 2, 0))
        grid = grid * std + mean
        grid = np.clip(grid, 0, 1)
        plt.figure(figsize=(12, 3))
        plt.imshow(grid)
        plt.title(classes.numpy())
        plt.axis("off")
        plt.show()


class HTSCUniformDataset:

    def __init__(self, root, batch_size=32, num_workers=8):
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        image_mean = processor.image_mean
        image_std = processor.image_std
        size = processor.size["height"]

        train_transforms = T.Compose(
            [
                T.RandomResizedCrop((size, size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std),
            ]
        )
        val_transforms = T.Compose(
            [
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std),
            ]
        )

        data_transforms = {
            "train": train_transforms,
            "val": val_transforms,
        }

        self.datasets = {
            x: torchvision.datasets.ImageFolder(f"{root}/{x}", data_transforms[x])
            for x in ["train", "val"]
        }

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=batch_size,
                shuffle=(x == "train"),
                num_workers=num_workers,
            )
            for x in ["train", "val"]
        }


if __name__ == "__main__":
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    print(image_mean, image_std)
