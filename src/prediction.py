import matplotlib.pyplot as plt
import torch

from omegaconf import OmegaConf

from .model_building import ModelBuilder
from .utils import set_device


NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
NORM_STD = torch.tensor([0.229, 0.224, 0.225])


class SceneSegmentor:
    def __init__(self,
                 config: OmegaConf,
                 test_generator: torch.utils.data.DataLoader,
                 visualize: bool = True,
                 to_visualize_samples: int = None):
        self.config = config
        self.device = set_device()
        self.model = ModelBuilder(self.config).build_model().to(self.device)
        self.test_loader = test_generator
        self.visualize = visualize
        self.to_visualize_samples = to_visualize_samples

    def run(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for test_batch in self.test_loader:
                test_images = test_batch[0].to(self.device).float()
                pr_masks = self._predict(test_images)
        if self.visualize:
            self._visualize(test_images, pr_masks)

    def _predict(self, test_images: torch.Tensor) -> torch.Tensor:
        test_outputs = self.model(test_images)  # Logits (Batch, Classes, H, W)
        # Convert logits to predicted class labels
        pr_masks = test_outputs.softmax(dim=1).argmax(dim=1)  # Shape: [batch_size, H, W]
        return pr_masks

    def _visualize(self, images: torch.Tensor, pr_masks: torch.Tensor) -> None:

        # Visualize a few samples (image and predicted mask)
        for idx, (image, pr_mask) in enumerate(zip(images, pr_masks)):
            if self.to_visualize_samples is not None and idx > self.to_visualize_samples:
                break

            print('Visualizing sample:', idx)
            plt.figure(figsize=(12, 6))

            # Original Image
            plt.subplot(1, 2, 1)
            # Denormalize the image tensor for display
            img_to_show_normalized = image.cpu() # Get a single image tensor and move to CPU
            # NORM_MEAN and NORM_STD should be reshaped to (3, 1, 1) for broadcasting across channels
            img_to_show_denormalized = img_to_show_normalized * NORM_STD.view(3, 1, 1) + NORM_MEAN.view(3, 1, 1)
            # Clamp values to [0, 1] range to ensure valid display by matplotlib
            img_to_show_clamped = torch.clamp(img_to_show_denormalized, 0, 1)
            # Permute from (C, H, W) to (H, W, C) for matplotlib
            img_to_show_np = img_to_show_clamped.permute(1, 2, 0).numpy()
            plt.imshow(img_to_show_np)
            plt.title("Image")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 2, 2)
            plt.imshow(pr_mask.cpu().numpy(), cmap="tab20")  # Visualize predicted mask
            plt.title("Prediction")
            plt.axis("off")

            # Show the figure
            plt.savefig(f"visualization_{idx}.png")
            plt.close()
