import numpy as np
import torch

from omegaconf import OmegaConf

from .model_building import ModelBuilder


def color_to_class(config: OmegaConf) -> dict:
    """
    Maps a color (RGB) to its corresponding class index as follows:
    COLOR_MAP = {(np.uint8(0), np.uint8(0), np.uint8(0)): 0,  # Background
                 (np.uint8(0), np.uint8(128), np.uint8(0)): 1,  # Tree
                 (np.uint8(64), np.uint8(0), np.uint8(128)): 2,  # Moving car
                 (np.uint8(64), np.uint8(64), np.uint8(0)): 3,  # Human
                 (np.uint8(128), np.uint8(0), np.uint8(0)): 4,  # Building
                 (np.uint8(128), np.uint8(64), np.uint8(128)): 5,  # Road
                 (np.uint8(128), np.uint8(128), np.uint8(0)): 6,  # Low vegetation
                 (np.uint8(192), np.uint8(0), np.uint8(192)): 7}  # Static car
    Returns:
        COLOR_MAP: {(R,G,B): class_index}
    """
    class_to_id = config.dataset.classes
    color_map_cfg = config.dataset.color_map

    COLOR_MAP = {}

    for class_name, class_idx in class_to_id.items():
        rgb = color_map_cfg[class_name]
        rgb_tuple = tuple(np.uint8(c) for c in rgb)

        COLOR_MAP[rgb_tuple] = class_idx

    return COLOR_MAP


def class_to_color(config: OmegaConf) -> dict:
    """
    Maps a class index to its corresponding color (RGB).
    Returns:
        CLASS_TO_COLOR: {class_index: np.array([R,G,B])}
    """
    class_to_id = config.dataset.classes
    color_map_cfg = config.dataset.color_map

    CLASS_TO_COLOR = {}

    for class_name, class_idx in class_to_id.items():
        rgb = color_map_cfg[class_name]
        rgb_tuple = tuple(np.uint8(c) for c in rgb)

        CLASS_TO_COLOR[class_idx] = np.array(rgb_tuple, dtype=np.uint8)

    return CLASS_TO_COLOR


def get_config_value(config, dotted_key: str):
    """Reads nested values like training.batch_size from OmegaConf."""
    try:
        return OmegaConf.select(config, dotted_key)
    except Exception:
        raise KeyError(f"Config key '{dotted_key}' not found.")


def load_model(model_path: str,
               config: OmegaConf,
               device: torch.device):
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
    # Setting device
    # try:
    #     device = torch.device("cuda:0")
    #     print('run with gpu')
    # except:
    device = torch.device("cpu")
    print(f"Using {device} device")
    return device
