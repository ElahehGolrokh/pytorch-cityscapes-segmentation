import cv2
import numpy as np
import os

from omegaconf import OmegaConf
from pathlib import Path


class Preprocessor():
    def __init__(self,
                 config: OmegaConf,
                 image_path: Path,
                 normalize_flag: bool = False,
                 mean: tuple = None,
                 std: tuple = None):
        self.image_path = image_path
        self.config = config
        self.normalize_flag = normalize_flag
        self.mean = mean
        self.std = std

    def preprocess_image(self):
        image = self._read_image(self.image_path)
        image = image.astype(float)
        if self.normalize_flag:
            image = self._normalize(image)
        return image
    
    @staticmethod
    def _read_image(file_path):
        image = np.load(file_path)
        return image

    def _normalize(self, image):
        """
        Standard normalization is applied using the formula:
        img = (img - mean * max_pixel_value) / (std * max_pixel_value).
        """
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        image = (image - mean) / std
        return image
