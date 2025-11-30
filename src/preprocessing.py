import cv2
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path


class Preprocessor():
    """
    Preprocess images for the model.

    Parameters:
    -----------
        config (OmegaConf): The configuration object.
        image_path (Path): The path to the image file.
        normalize_flag (bool): Whether to normalize the image.
        mean (tuple): The mean values for normalization.
        std (tuple): The standard deviation values for normalization.

    Private_Methods:
    -----------------
        _read_image
        _normalize

    Public_Methods:
    ----------------
        preprocess_image

    Example:
    --------
    >>> preprocessor = Preprocessor(config,
                                    image_path,
                                    normalize_flag=True,
                                    mean=(0.5,),
                                    std=(0.5,))
    >>> processed_image = preprocessor.preprocess_image()
    """
    def __init__(self,
                 config: OmegaConf,
                 normalize_flag: bool = False,
                 mean: tuple = None,
                 std: tuple = None):
        self.config = config
        self.normalize_flag = normalize_flag
        self.mean = mean
        self.std = std

    def preprocess_image(self,
                         image_path: Path = None,
                         image: np.ndarray = None) -> np.ndarray:
        """Run preprocessing on the image."""
        if image is None:
            try:
                image = self._read_image(image_path)
            except Exception as e:
                raise ValueError(f"Error reading image from {image_path}: {e}") from e
        image = image.astype(float)
        if self.normalize_flag:
            image = self._normalize(image)
        return image

    @staticmethod
    def _read_image(file_path: Path) -> np.ndarray:
        """Read an image from a file."""
        if file_path.endswith('.npy'):
            image = np.load(file_path)
        else:
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image/255)
        return image

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Standard normalization is applied using the formula:
        img = (img - mean) / (std).
        """
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        image = (image - mean) / std
        return image
