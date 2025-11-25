import matplotlib.pyplot as plt
import numpy as np
import torch

from omegaconf import OmegaConf
from pathlib import Path

from .utils import class_to_color, decode_mask_for_plot, \
    denormalize_image, load_model, set_device


class SceneSegmentor:
    def __init__(self,
                 config: OmegaConf,
                 model_path: Path,
                 test_generator: torch.utils.data.DataLoader,
                 visualize: bool = True,
                 to_visualize_samples: int = None):
        self.config = config
        self.model_path = model_path
        self.test_loader = test_generator
        self.visualize = visualize
        self.to_visualize_samples = to_visualize_samples

    def run(self) -> None:
        self._setup()
        self.model.eval()
        with torch.no_grad():
            for test_batch in self.test_loader:
                test_images = test_batch[0].to(self.device).float()
                pr_masks = self._predict(test_images)
        if self.visualize:
            self._visualize(test_images, pr_masks)
    
    def _setup(self):
        self.device = set_device()
        self.model = load_model(self.model_path, self.config, self.device)

    def _predict(self, test_images: torch.Tensor) -> torch.Tensor:
        test_outputs = self.model(test_images)  # Logits (Batch, Classes, H, W)
        # Convert logits to predicted class labels
        pr_masks = test_outputs.softmax(dim=1).argmax(dim=1)  # Shape: [batch_size, H, W]
        return pr_masks

    def _visualize(self, images: torch.Tensor, pr_masks: torch.Tensor) -> None:

        for idx, (image, pr_mask) in enumerate(zip(images, pr_masks)):
            if self.to_visualize_samples is not None and idx > self.to_visualize_samples:
                break

            # post-process the image
            denormalized_image = denormalize_image(image)
            # post-process the predicted mask
            CLASS_TO_COLOR = class_to_color(self.config)
            rgb_pred = decode_mask_for_plot(pr_mask.cpu().numpy(),
                                            CLASS_TO_COLOR)
            
            plt.figure(figsize=(12, 6))
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(denormalized_image)
            plt.title("Image")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 2, 2)
            plt.imshow(rgb_pred)
            plt.title("Prediction")
            plt.axis("off")
            
            logs_dir = self.config.dirs.logs
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{logs_dir}/visualization_{idx}.png")
            plt.close()
