import cv2
import numpy as np
import os
import torch
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from albumentations import (HorizontalFlip, ShiftScaleRotate, Resize, Compose, ToTensorV2)
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .preprocessing import Preprocessor


def create_data_paths(dir):
    data_paths = []
    for seq in os.listdir(dir):
        seq_dir_images = os.path.join(dir, seq, 'Images')
        for files in os.listdir(seq_dir_images):
            if files.endswith('.jpg') or files.endswith('.png'):
                file_path = os.path.join(seq_dir_images, files)
                data_paths.append(file_path)
    return data_paths

def get_transforms(phase, height, width):
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
#                 GaussNoise(),
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
    def __init__(self,
                 config: OmegaConf,
                 data_paths: Path,
                 phase: str):
        self.config = config
        self.data_paths = data_paths
        self.mean = tuple(config.dataset.mean)
        self.std = tuple(config.dataset.std)
        self.height = config.dataset.height
        self.width = config.dataset.width
        self.transforms = get_transforms(phase, self.height, self.width)
        self.phase = phase

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data_paths[idx]
        preprocessor = Preprocessor(config=self.config,
                                    image_path=image_path,
                                    normalize_flag=True,
                                    mean=self.mean,
                                    std=self.std)
        image = preprocessor.preprocess_image()

        if self.phase == "test":
            # image = image.unsqueeze(0)  # Add batch dimension
            # image = np.expand_dims(image, axis=0)
            # mask = torch.zeros(1, image.shape[2], image.shape[3], dtype=torch.long)
            mask = np.zeros((image.shape[0], image.shape[1]))
        else:
            mask = preprocessor.preprocess_mask()
            # Removed: mask = mask.astype(float) to keep mask as integer type

        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        
        # Explicitly convert mask to LongTensor for CrossEntropyLoss
        mask = augmented['mask'].long()

        return image, mask


class DataGenerator:
    def __init__(self,
                 config: OmegaConf,
                 phase: str,
                 batch_size: int,
                 shuffle: bool):
        self.config = config
        self.phase = phase
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self, paths: list):
        dataset = SemanticSegmentationDataset(data_paths=paths,
                                              phase=self.phase,
                                              config=self.config)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=os.cpu_count())
        return dataloader
