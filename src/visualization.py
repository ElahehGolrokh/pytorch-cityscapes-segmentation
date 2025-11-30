import matplotlib.pyplot as plt
import numpy as np
import torch

from omegaconf import OmegaConf
from pathlib import Path

from .utils import load_image, mask_to_rgb


class SegmentationVisualizer:
    def __init__(self,
                 config: OmegaConf,
                 images_path: list[Path],
                 logs_dir: Path,
                 pr_masks: torch.Tensor,
                 to_visualize_samples: int):
        self.config = config
        self.images_path = images_path
        self.logs_dir = logs_dir
        self.pr_masks = pr_masks
        self.to_visualize_samples = to_visualize_samples

    def visualize(self) -> None:
        """
        Visualize the original images and their predicted masks.
        Save the visualizations to the logs directory defined in the config.
        """
        color_map = self.config.dataset.color_map
        for idx, (image_path, pr_mask) in enumerate(zip(self.images_path, self.pr_masks)):
            if self.to_visualize_samples is not None and idx > self.to_visualize_samples:
                break

            image = load_image(image_path)
            rgb_frame = mask_to_rgb(np.array(pr_mask, dtype=np.uint8), color_map)

            plt.figure(figsize=(12, 6))
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 2, 2)
            plt.imshow(rgb_frame)
            plt.title("Prediction")
            plt.axis("off")

            plt.savefig(f"{self.logs_dir}/visualization_{idx}.png")
            plt.close()
