import torch

from omegaconf import OmegaConf
from pathlib import Path

from .memory_utils import MemoryMonitor
from .utils import load_model, set_device


class SceneSegmentor:
    """
    Segmentor for scene images. This class is responsible for segmenting
    images into pre-defined classes using a pre-trained model.

    Parameters:
    -----------
        config (OmegaConf): The configuration object.
        model_path (Path): The path to the pre-trained model.
        test_generator (DataLoader): The data loader for the test dataset.

    Attributes:
    -----------
        device_ (torch.device): The device to run the model on.
        model_ (torch.nn.Module): The pre-trained model for segmentation.

    Private_Methods:
    ---------------
        _setup
        _predict

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
                 memory_threshold: float = None,
                 check_interval: int = 5):
        self.config = config
        self.model_path = model_path
        self.test_loader = test_generator

        # Optional memory monitoring
        self.memory_monitor = None
        if memory_threshold is not None:
            self.memory_monitor = MemoryMonitor(
                threshold=memory_threshold,
                check_interval=check_interval  # Check every 5 batches
            )

    def run(self) -> torch.Tensor:
        """Run the segmentation on the test dataset."""
        try:
            self._setup()
            self.model_.eval()
            all_masks = []
            with torch.no_grad():
                for batch_idx, test_batch in enumerate(self.test_loader):
                    test_images = test_batch[0].to(self.device_).float()
                    pr_masks = self._predict(test_images)
                    all_masks.append(pr_masks.cpu())
                    # Check memory if monitoring is enabled
                    if self.memory_monitor is not None:
                        try:
                            self.memory_monitor.check()
                        except MemoryError as e:
                            print(f"\n⚠️  Memory limit reached during inference (batch {batch_idx + 1})")
                            raise
            return torch.cat(all_masks, dim=0)
        except MemoryError as e:
            raise MemoryError(f"\n❌ Processing stopped due to memory constraints"
                              f"\n    {str(e)}"
                              f"\nSuggestion: Try reducing video resolution or processing in chunks")
        except Exception as e:
            raise RuntimeError(f"\n❌ Error during processing: {e}")

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
