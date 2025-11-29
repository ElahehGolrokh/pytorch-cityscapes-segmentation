import matplotlib.pyplot as plt
import torch

from omegaconf import OmegaConf
from pathlib import Path

from .utils import load_image


class SegmentationVisualizer:
    def __init__(self,
                 config: OmegaConf,
                 images_path: list[Path],
                 pr_masks: torch.Tensor,
                 to_visualize_samples: int):
        self.config = config
        self.images_path = images_path
        self.pr_masks = pr_masks
        self.to_visualize_samples = to_visualize_samples

    def visualize(self) -> None:
        """
        Visualize the original images and their predicted masks.
        Save the visualizations to the logs directory defined in the config.
        """
        for idx, (image_path, pr_mask) in enumerate(zip(self.images_path, self.pr_masks)):
            if self.to_visualize_samples is not None and idx > self.to_visualize_samples:
                break

            image = load_image(image_path)

            # post-process the image
            # denormalized_image = denormalize_image(self.config, image)

            plt.figure(figsize=(12, 6))
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 2, 2)
            plt.imshow(pr_mask)
            plt.title("Prediction")
            plt.axis("off")

            logs_dir = self.config.dirs.logs
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{logs_dir}/visualization_{idx}.png")
            plt.close()
