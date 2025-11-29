import cv2
import numpy as np
import os
import pandas as pd
import torch
# Ignore warnings
import warnings

from albumentations import (HorizontalFlip, ShiftScaleRotate,
                            Resize, Compose, ToTensorV2)
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .preprocessing import Preprocessor


warnings.filterwarnings("ignore")


def create_data_paths(dir: Path,
                      df: pd.DataFrame = None) -> list[Path]:
    """Creates a list of image files from the dataset directory."""
    if df is None:
        paths = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
    else:
        df = pd.DataFrame(data=[os.path.join(dir, df['image_path'].values[i]) for i in range(len(df))],
                          columns=['image_path'])
        paths = df['image_path'].values.tolist()
    return paths


def encode_labels(mask: np.ndarray, mapping_dict: dict) -> np.ndarray:
    """
    Encodes the labels in the mask using the mapping dictionary.
    This function converts the original label values (-1 to 19) to the new
    values defined in the mapping dictionary from 0 to 10.
    """
    label_mask = np.zeros_like(mask)
    for k in mapping_dict.keys():
        label_mask[mask == k] = mapping_dict[k]
    return label_mask


def get_transforms(phase: str,
                   height: int,
                   width: int) -> Compose:
    """Returns the data augmentation using Albumentations."""
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                # GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Resize(height, width),
            ToTensorV2()
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


class SemanticSegmentationDataset(Dataset):
    """
    Custom dataset for semantic segmentation.

    Parameters:
    -----------
        config (OmegaConf): The configuration object.
        mean : float, optional
            Overrides mean in config if provided.
        std : float, optional
            Overrides std in config if provided.
        height : int, optional
            Overrides height in config if provided.
        width : int, optional
            Overrides width in config if provided.
        data_paths (Path): The paths to the image files.
        phase (str): The phase of the data preparation (train, val, test).


    """
    def __init__(self,
                 config: OmegaConf,
                 phase: str,
                 data_paths: list = None,
                 video_frame: np.ndarray = None,
                 mean: float = None,
                 std: float = None,
                 height: int = None,
                 width: int = None):
        self.config = config

        # Parameters from config
        self.mean = tuple(config.dataset.mean) if mean is None else (mean,)
        self.std = tuple(config.dataset.std) if std is None else (std,)
        self.height = config.dataset.height if height is None else height
        self.width = config.dataset.width if width is None else width
        self.mapping_dict = config.dataset.mapping_dict

        # Data paths and transformations
        self.data_paths = data_paths
        self.video_frame = video_frame
        self.transforms = get_transforms(phase, self.height, self.width)
        self.phase = phase

    def __len__(self):
        if self.data_paths is not None:
            return len(self.data_paths)
        elif self.video_frame is not None:
            return len(self.video_frame)
        else:
            raise ValueError("Invalid dataset input.")

    def __getitem__(self, idx):
        """Fetches image, mask pair for the given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.data_paths is not None:
            # image processing
            image_path = self.data_paths[idx]
            preprocessor = Preprocessor(config=self.config,
                                        image_path=image_path,
                                        normalize_flag=True,
                                        mean=self.mean,
                                        std=self.std)
            image = preprocessor.preprocess_image()
            mask_path = image_path.replace('image', 'label')
            if 'image' in mask_path and Path(mask_path).exists():
                mask = np.load(mask_path)
                # Removed: mask = mask.astype(float) to keep mask as integer type
            else:
                mask = np.zeros((image.shape[0], image.shape[1]),
                                dtype=np.uint8)
        else:
            # video frame processing
            preprocessor = Preprocessor(config=self.config,
                                        normalize_flag=True,
                                        mean=self.mean,
                                        std=self.std)
            video_frame = self.video_frame[idx]
            image = preprocessor.preprocess_image(video_frame)
            mask = np.zeros((image.shape[0], image.shape[1]),
                            dtype=np.uint8)

        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        # Apply remapping after augmentations and ToTensorV2
        mask = encode_labels(mask, self.mapping_dict)
        # Explicitly convert mask to LongTensor for CrossEntropyLoss
        mask = torch.Tensor(mask).long()

        return image, mask


class DataGenerator:
    """
    Data generator for training, validation, and testing phases.

    Parameters:
    -----------
        config (OmegaConf): The configuration object.
        phase (str): The phase of the data preparation (train, val, test).
        batch_size (int): The batch size for data loading.
        shuffle (bool): Whether to shuffle the data.

    Public Methods:
    ----------------
        load_data: Loads data for the specified phase.

    Example:
    >>> generator = DataGenerator(config, phase="train", batch_size=32, shuffle=True)
    >>> dataloader = generator.load_data(train_image_paths)
    """
    def __init__(self,
                 config: OmegaConf,
                 phase: str,
                 batch_size: int,
                 shuffle: bool):
        self.config = config
        self.phase = phase
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self,
                  paths: list = None,
                  video_frame: np.ndarray = None) -> DataLoader:
        """
        Loads data for the specified phase.
        """
        dataset = SemanticSegmentationDataset(data_paths=paths,
                                              phase=self.phase,
                                              video_frame=video_frame,
                                              config=self.config)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=os.cpu_count())
        return dataloader
