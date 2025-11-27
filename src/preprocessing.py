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
                 image_path: Path,
                 normalize_flag: bool = False,
                 mean: tuple = None,
                 std: tuple = None):
        self.image_path = image_path
        self.config = config
        self.normalize_flag = normalize_flag
        self.mean = mean
        self.std = std

    def preprocess_image(self) -> np.ndarray:
        """Run preprocessing on the image."""
        image = self._read_image(self.image_path)
        image = image.astype(float)
        if self.normalize_flag:
            image = self._normalize(image)
        return image

    @staticmethod
    def _read_image(file_path: Path) -> np.ndarray:
        """Read an image from a file."""
        image = np.load(file_path)
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
