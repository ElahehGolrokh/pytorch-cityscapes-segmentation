import cv2
import numpy as np
import os

from omegaconf import OmegaConf
from pathlib import Path

from .utils import color_to_class


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
        if self.normalize_flag:
            image = self._normalize(image)
        return image

    def preprocess_mask(self):
        mask_path = self.image_path.replace('Images', 'Labels')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Label file not found for: {mask_path}")
        mask = self._read_image(mask_path)
        mask = self._rgb_to_gray(mask)
        return mask
    
    @staticmethod
    def _read_image(file_path):
        image = cv2.imread(file_path)  # Loads as BGR, numpy array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype(float)
        return image

    def _normalize(self, image):
        """
        Standard normalization is applied using the formula:
        img = (img - mean * max_pixel_value) / (std * max_pixel_value).
        """
        image = image/255.
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        image = (image - mean) / std
        return image

    def _rgb_to_gray(self, rgb_label_array):
        """
        Converts a 3-channel RGB label image to a single-channel class ID mask.

        Args:
            rgb_label_image (np.ndarray): The 3-channel (H, W, 3) RGB label image.

        Returns:
            np.ndarray: A single-channel (H, W) mask with integer class IDs.
        """
        color_map = color_to_class(self.config)
        rgb_pixels_tuples = [tuple(p) for p in rgb_label_array.reshape(-1, 3)]
        gray_mask = [color_map[rgb_pixels_tuples[i]] for i in range(len(rgb_pixels_tuples))]
        gray_mask = np.array(gray_mask, dtype=np.uint8).reshape(rgb_label_array.shape[0],
                                                                rgb_label_array.shape[1])
        return gray_mask
