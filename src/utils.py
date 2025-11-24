from omegaconf import OmegaConf


def get_config_value(config, dotted_key: str):
    """Reads nested values like training.batch_size from OmegaConf."""
    try:
        return OmegaConf.select(config, dotted_key)
    except Exception:
        raise KeyError(f"Config key '{dotted_key}' not found.")
