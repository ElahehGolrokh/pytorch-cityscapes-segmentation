import numpy as np
import segmentation_models_pytorch as smp
import torch

from omegaconf import OmegaConf
from pathlib import Path

from .model_building import ModelBuilder
from .utils import set_device


class Trainer:
    """
    Trainer class for training the segmentation model.

    Parameters
    ----------
    config : OmegaConf
        Configuration object containing training parameters.
    train_generator : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_generator : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Initial learning rate for the optimizer.
    loss : str
        Loss function's name to be used during training.
    num_classes : int
        Number of classes in the segmentation dataset.

    Attributes
    ----------
    best_metric_ : float
        Best value of the evaluation metric (e.g., IoU) achieved during training.
    best_metric_epoch_ : int
        Epoch at which the best_metric_ was achieved.
    metric_values_ : list
        List to store the values of the evaluation metric (e.g., IoU) for each epoch.
    epoch_loss_values_ : list
        List to store the loss values for each epoch.
    device_ : torch.device
        Device on which the model is trained (CPU or GPU).
    loss_function_ : torch.nn.Module
        Loss function used during training.
    model_ : torch.nn.Module
        Segmentation model.
    optimizer_ : torch.optim.Optimizer
        Optimizer for training the model.
    scheduler_ : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler.

    Private_Methods
    ----------------
    _setup
    _check_dir
    _save_model
    _validate

    Public_Methods
    ---------------
    fit

    Example:
    ---------
    >>> trainer = Trainer(config, train_generator, val_generator)
    >>> trainer.fit()

    """
    def __init__(self,
                 config: OmegaConf,
                 train_generator: torch.utils.data.DataLoader,
                 val_generator: torch.utils.data.DataLoader,):
        self.config = config
        self.train_generator = train_generator
        self.val_generator = val_generator

        # Setting up model parameters from config
        self.epochs = self.config.training.epochs
        self.learning_rate = self.config.training.initial_learning_rate
        self.loss = self.config.training.loss
        self.num_classes = len(self.config.dataset.classes)
        self.run_dir = Path(self.config.dirs.run)

        # Placeholders
        # Initialize best_metric with a value that will be easily surpassed
        self.best_metric_ = -1
        self.best_metric_epoch_ = -1
        self.metric_values_ = []   # This will store average IoU for each epoch
        self.epoch_loss_values_ = []
        self.device_: torch.device = None
        self.loss_function_: torch.nn.Module = None
        self.model_: torch.nn.Module = None
        self.optimizer_: torch.optim.Optimizer = None
        self.scheduler_: torch.optim.lr_scheduler.LambdaLR = None

    def _setup(self) -> None:
        """
        Setup the trainer by checking directories and setting the
        device, the model, and other components.
        """
        self._check_dir()

        # Setting device
        self.device_ = set_device()

        # Build the model
        model_builder = ModelBuilder(self.config)
        self.model_ = model_builder.build_model().to(self.device_)
        self.model_ = self.model_.float()

        if self.loss == "cross_entropy":
            self.loss_function_ = torch.nn.CrossEntropyLoss()
        elif self.loss == "focal_loss":
            self.loss_function_ = smp.losses.FocalLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

        self.optimizer_ = torch.optim.Adam(self.model_.parameters(),
                                           self.learning_rate,
                                           weight_decay=0)
        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_,
                                                                     self.epochs)

    def _check_dir(self) -> None:
        """
        Check if the run directory exists, if not, create it.
        """
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True)
            print(f"Created directory: {self.run_dir}")

    def _save_model(self, epoch: int, avg_val_iou: float) -> None:
        """
        Save the model checkpoint.
        """
        model_path = self.run_dir / f'best_model_epoch{epoch+1}_{avg_val_iou:.4f}.pth'
        torch.save(self.model_.state_dict(), model_path)
        print(f"###### Congratulations ###### saved new model to {model_path}")

    def fit(self,) -> None:
        """
        Start the training process.
        """
        self._setup()
        for epoch in range(self.epochs):
            print('-' * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model_.train()
            epoch_loss = 0
            step = 0
            for batch_data in self.train_generator:
                step += 1
                inputs, labels = batch_data[0].to(self.device_).float(), batch_data[1].to(self.device_)
                self.optimizer_.zero_grad()
                outputs = self.model_(inputs)  # Logits (Batch, Classes, H, W)
                loss = self.loss_function_(outputs, labels)  # CrossEntropyLoss expects logits and Long labels
                loss.backward()
                self.optimizer_.step()
                epoch_loss += loss.item()

            epoch_loss /= step
            self.epoch_loss_values_.append(epoch_loss)

            # Validation
            self._validate(epoch)
            # Use get_last_lr() for PyTorch 1.4+ and later versions
            print('\n', f"###### epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"train completed, best_metric: {self.best_metric_:.4f} at epoch: {self.best_metric_epoch_}")

    def _validate(self, epoch: int) -> None:
        """
        Validate the model on the validation set in each epoch.
        """
        self.model_.eval()
        with torch.no_grad():
            val_iou_scores = []
            for val_data_batch in self.val_generator:
                val_images, val_labels = val_data_batch[0].to(self.device_).float(), val_data_batch[1].to(self.device_)
                val_outputs = self.model_(val_images)  # Logits (Batch, Classes, H, W)

                # Convert logits to predicted class labels
                pr_masks = val_outputs.softmax(dim=1).argmax(dim=1)  # Shape: [batch_size, H, W]

                tp, fp, fn, tn = smp.metrics.get_stats(
                    pr_masks,  # Predicted mask (argmaxed)
                    val_labels,  # Ground truth mask
                    mode='multiclass',
                    num_classes=self.num_classes,  # Number of classes (including background)
                    ignore_index=None,  # Set an index if you want to ignore a specific class
                )

                batch_iou = smp.metrics.iou_score(
                    tp, fp, fn, tn,
                    reduction='macro'
                )
                val_iou_scores.append(batch_iou.item())

            avg_val_iou = np.mean(val_iou_scores)
            # Store the average IoU for this epoch
            self.metric_values_.append(avg_val_iou)

            if avg_val_iou > self.best_metric_:
                self.best_metric_ = avg_val_iou
                self.best_metric_epoch_ = epoch + 1
                self._save_model(epoch, avg_val_iou)
            else:
                self.scheduler_.step()
                print('The learning rate updated to :', self.scheduler_.get_last_lr()[0])
            print(f"current epoch: {epoch + 1} current IoU: {avg_val_iou:.4f}"
                  f" best IoU: {self.best_metric_:.4f} at epoch: {self.best_metric_epoch_}")
