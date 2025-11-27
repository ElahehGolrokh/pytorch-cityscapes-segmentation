import matplotlib.pyplot as plt
import torch

from omegaconf import OmegaConf
from pathlib import Path

from .utils import denormalize_image, load_model, set_device


class SceneSegmentor:
    """
    Segmentor for scene images. This class is responsible for segmenting
    images into pre-defined classes using a pre-trained model.

    Parameters:
    -----------
        config (OmegaConf): The configuration object.
        model_path (Path): The path to the pre-trained model.
        test_generator (DataLoader): The data loader for the test dataset.
        visualize (bool): Whether to visualize the predictions.
        to_visualize_samples (int): Number of samples to visualize.

    Attributes:
    -----------
        device_ (torch.device): The device to run the model on.
        model_ (torch.nn.Module): The pre-trained model for segmentation.

    Private_Methods:
    ---------------
        _setup
        _predict
        _visualize

    Public_Methods:
    ---------------
        run

    Example:
    >>> segmentor = SceneSegmentor(config, model_path, test_generator)
    >>> predictions = segmentor.run()
    """
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

    def run(self) -> torch.Tensor:
        """Run the segmentation on the test dataset."""
        self._setup()
        self.model_.eval()
        with torch.no_grad():
            for test_batch in self.test_loader:
                test_images = test_batch[0].to(self.device_).float()
                pr_masks = self._predict(test_images)
        if self.visualize:
            self._visualize(test_images, pr_masks)
        return pr_masks

    def _setup(self) -> None:
        """Set up the device and model for inference."""
        self.device_ = set_device()
        self.model_ = load_model(self.model_path, self.config, self.device_)

    def _predict(self, test_images: torch.Tensor) -> torch.Tensor:
        """Run the model on the test images and return the predicted masks."""
        # Logits (Batch, Classes, H, W)
        test_outputs = self.model_(test_images)
        # Convert logits to predicted class labels; Shape: [batch_size, H, W]
        pr_masks = test_outputs.softmax(dim=1).argmax(dim=1)
        return pr_masks

    def _visualize(self, images: torch.Tensor, pr_masks: torch.Tensor) -> None:
        """
        Visualize the original images and their predicted masks.
        Save the visualizations to the logs directory defined in the config
        if `self.visualize` is True.
        """
        for idx, (image, pr_mask) in enumerate(zip(images, pr_masks)):
            if self.to_visualize_samples is not None and idx > self.to_visualize_samples:
                break

            # post-process the image
            denormalized_image = denormalize_image(self.config, image)

            plt.figure(figsize=(12, 6))
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(denormalized_image)
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
