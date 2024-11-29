import os
import sys
import glob
import time

import numpy as np
from scipy.special import softmax
import pandas as pd
import cv2
from sklearn.metrics import classification_report
from tqdm import tqdm

import onnx
import onnxruntime as ort

import models


class ONNXModel:
    def __init__(self, model_path):
        self.model_path = model_path

        self.mean = np.array([0.5] * 3, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5] * 3, dtype=np.float32).reshape(1, 1, 3)
        self.load_model()

    def load_model(self):
        self.gpu_session = ort.InferenceSession(
            self.model_path,
            providers=["CUDAExecutionProvider"],
        )
        self.cpu_session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.gpu_session.get_inputs()[0].name

        # Warm up
        print("Warming up ...")
        for _ in range(10):
            dummy_image = np.random.randn(2, 3, 224, 224).astype(np.float32)
            self.gpu_session.run(None, {self.input_name: dummy_image})
            self.cpu_session.run(None, {self.input_name: dummy_image})

    def preprocess(self, image_path):
        img: np.ndarray = cv2.imread(image_path, flags=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.pad_to_aspect_ratio(img)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img = img / 255
        img = (img - self.mean) / self.std
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        return img

    def pad_to_aspect_ratio(
        self,
        img: np.ndarray,
        aspect_ratio=0.75,
        fill_color=255,
    ):
        h, w = img.shape[:2]
        target_w, target_h = w, h

        if w / h < aspect_ratio:  # Image is too tall, pad width
            target_w = int(h * aspect_ratio)
        elif w / h > aspect_ratio:  # Image is too wide, pad height
            target_h = int(w / aspect_ratio)

        pad_w = (target_w - w) // 2
        pad_h = (target_h - h) // 2

        # Define padding in (top, bottom) and (left, right) order
        padding = (
            (pad_h, target_h - h - pad_h),
            (pad_w, target_w - w - pad_w),
            (0, 0),
        )
        # Fill with white
        img = np.pad(
            img,
            padding,
            mode="constant",
            constant_values=fill_color,
        )
        return img

    def inference(self, image_path):
        img = self.preprocess(image_path)
        ort_inputs = {self.input_name: img}
        logit: np.ndarray = self.gpu_session.run(None, ort_inputs)[0]
        return logit.argmax(1)[0], softmax(logit)

    def evaluate_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.reset_index()  # make sure indexes pair with number of rows
        gpu_preds, cpu_preds = [], []
        gpu_time, cpu_time = 0, 0

        for index, row in tqdm(df.iterrows()):
            image_path = row["path"]
            image = self.preprocess(image_path)

            ort_inputs = {self.input_name: image}

            start_time = time.time()
            logit: np.ndarray = self.gpu_session.run(None, ort_inputs)[0]
            gpu_time += time.time() - start_time
            gpu_preds.append(logit.argmax(1)[0])

            start_time = time.time()
            logit: np.ndarray = self.cpu_session.run(None, ort_inputs)[0]
            cpu_time += time.time() - start_time
            cpu_preds.append(logit.argmax(1)[0])

        gpu_report = classification_report(df["label"].values, gpu_preds)
        cpu_report = classification_report(df["label"].values, cpu_preds)

        print(gpu_report)
        print(cpu_report)
        print("Average GPU time:", (gpu_time / len(df.index)) * 1000)
        print("Average CPU time:", (cpu_time / len(df.index)) * 1000)


if __name__ == "__main__":
    onnx_path = "mobilenet_v3_softmax.onnx"

    if not os.path.exists(onnx_path):
        print("File doesn't exist")
        model = models.MobileNetV3.load_from_checkpoint(
            "bidv_uniform_classification/icuw0hhe/checkpoints/best.ckpt"
        )
        onnx_path = model.to_onnx_fp16(
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
        )
    else:
        print("File already exists")
        root, ext = os.path.splitext(onnx_path)
        onnx_path = f"{root}_fp16{ext}"

    onnx_model = ONNXModel(onnx_path)
    # onnx_model.evaluate_csv("datasets/uniform_bidv/three_classes/test.csv")
    print(onnx_model.inference("datasets/photo_2024-11-14_17-59-49.jpg"))
    print(onnx_model.inference("datasets/photo_2024-11-14_17-59-50.jpg"))
