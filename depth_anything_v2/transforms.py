import cv2
import numpy as np


class Resize:
    def __init__(self, width, height, keep_aspect_ratio=False):
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, sample):
        image = sample["image"]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return {"image": image}


class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        image = sample["image"]
        image = (image - self.mean) / self.std
        return {"image": image}


class PrepareForNet:
    def __call__(self, sample):
        image = sample["image"]
        image = image.transpose(2, 0, 1)
        return {"image": image.astype(np.float32)}
