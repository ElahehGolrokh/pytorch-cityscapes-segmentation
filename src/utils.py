from pathlib import Path
import cv2
import numpy as np
import torch

from omegaconf import OmegaConf
from tqdm import tqdm

from .model_building import ModelBuilder


def denormalize_image(config: OmegaConf,
                      image: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes an image tensor using the specified mean and std.
    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
    Returns:
        torch.Tensor: The denormalized image tensor (C, H, W).
    """
    mean = torch.tensor(config.dataset.mean)
    std = torch.tensor(config.dataset.std)
    img_to_show_normalized = image.cpu()
    # mean and std should be reshaped to (3, 1, 1) for broadcasting across channels
    img_to_show_denormalized = img_to_show_normalized * std.view(3, 1, 1) + mean.view(3, 1, 1)
    # Clamp values to [0, 1] range to ensure valid display by matplotlib
    img_to_show_clamped = torch.clamp(img_to_show_denormalized, 0, 1)
    # Permute from (C, H, W) to (H, W, C) for matplotlib
    img_to_show_np = (img_to_show_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img_to_show_np


def mask_to_rgb(mask, color_map):
    """
    mask: (H, W) array of class IDs
    color_map: dict {class_id: (R, G, B)}
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        rgb[mask == class_id] = color  # color is (R, G, B)

    return rgb


def get_config_value(config: OmegaConf,
                     dotted_key: str) -> any:
    """Reads nested values like training.batch_size from OmegaConf."""
    try:
        return OmegaConf.select(config, dotted_key)
    except Exception:
        raise KeyError(f"Config key '{dotted_key}' not found.")


def load_model(model_path: str,
               config: OmegaConf,
               device: torch.device) -> torch.nn.Module:
    """
    Loads a model from a specified path.
    """
    # Re-instantiate the model with the correct architecture
    # Build the model
    model_builder = ModelBuilder(config)
    model = model_builder.build_model().to(device)

    # Load the saved state dictionary
    try:
        model.load_state_dict(torch.load(model_path,
                                         map_location=device))
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Error loading model state_dict: {e}\n"
                         f"Ensure the path is correct and the model architecture matches.")


def set_device() -> torch.device:
    """
    Sets the device for training (GPU or CPU).
    """
    # try:
    #     device = torch.device("cuda:0")
    #     print('run with gpu')
    # except:
    device = torch.device("cpu")
    print(f"Using {device} device")
    return device


def create_video_output(cap: cv2.VideoCapture,
                        config: OmegaConf,
                        predicted_masks: np.ndarray,
                        output_path: str = "output_video.avi") -> None:
    color_map = config.dataset.color_map
    frame_width = config.dataset.width
    frame_height = config.dataset.height
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # save predicted masks
    for i in tqdm(range(predicted_masks.shape[0])):
        color_frame = mask_to_rgb(np.array(predicted_masks, dtype=np.uint8)[i],
                                  color_map)
        output.write(color_frame)
    output.release()
