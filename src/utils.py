import torch

from omegaconf import OmegaConf

from .model_building import ModelBuilder


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
