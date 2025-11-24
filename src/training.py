import numpy as np
import segmentation_models_pytorch as smp
import torch

from omegaconf import OmegaConf
from pathlib import Path


class Trainer:
    def __init__(self,
                 config: OmegaConf,
                 train_generator: torch.utils.data.DataLoader,
                 val_generator: torch.utils.data.DataLoader,):
        self.config = config
        self.train_generator = train_generator
        self.val_generator = val_generator
        self._metric_values = []   # This will store average IoU for each epoch
        self._epoch_loss_values = []

    def _setup(self):
        # Setting up model parameters
        encoder_name = self.config.training.encoder_name
        self.epochs = self.config.training.epochs
        learning_rate = self.config.training.initial_learning_rate
        loss = self.config.training.loss
        self.num_classes = len(self.config.training.classes)
        self.run_dir = Path(self.config.dirs.run)
        self._check_dir()

        # Setting device
        # try:
        #     device = torch.device("cuda:0")
        #     print('run with gpu')
        # except:
        self.device = torch.device("cpu")
        print(f"Using {self.device} device")

        activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
        self.model = smp.Unet(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,                     # model output classes (number of classes in your dataset)
            activation=activation
        ).to(self.device)
        self.model = self.model.float()

        if loss == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss == "focal_loss":
            self.loss_function = smp.losses.FocalLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

    def _check_dir(self):
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True)
            print(f"Created directory: {self.run_dir}")

    def _save_model(self, epoch: int, avg_val_iou: float):
        model_path = self.run_dir / f'best_model_epoch{epoch}_{avg_val_iou:.4f}.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"###### Congratulations ###### saved new best metric model to {model_path}")

    def fit(self,):
        self._setup()
        # Initialize with a value that will be easily surpassed
        self.best_metric = -1
        self.best_metric_epoch = -1
        for epoch in range(self.epochs):
            print('-' * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0
            for batch_data in self.train_generator:
                step += 1
                inputs, labels = batch_data[0].to(self.device).float(), batch_data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # Logits (Batch, Classes, H, W)
                loss = self.loss_function(outputs, labels)  # CrossEntropyLoss expects logits and Long labels
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= step
            self._epoch_loss_values.append(epoch_loss)

            # Validation
            self._validate(epoch)
            # Use get_last_lr() for PyTorch 1.4+ and later versions
            print('\n', f"###### epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")

    def _validate(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            val_iou_scores = []
            for val_data_batch in self.val_generator:
                val_images, val_labels = val_data_batch[0].to(self.device).float(), val_data_batch[1].to(self.device)
                val_outputs = self.model(val_images) # Logits (Batch, Classes, H, W)

                # Convert logits to predicted class labels
                pr_masks = val_outputs.softmax(dim=1).argmax(dim=1)  # Shape: [batch_size, H, W]

                # Calculate TP, FP, FN, TN for multiclass
                # The `get_stats` function computes a confusion matrix for each image/batch
                # and returns summed tp, fp, fn, tn.
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pr_masks,  # Predicted mask (argmaxed)
                    val_labels,  # Ground truth mask
                    mode='multiclass',
                    num_classes=self.num_classes,  # Number of classes (including background)
                    ignore_index=None,  # Set an index if you want to ignore a specific class
                )

                # Calculate IoU score from the stats
                # The `reduction` parameter applies to how the IoU is averaged across classes and/or images.
                # 'micro' means sum TP, FP, FN across classes and then calculate IoU.
                batch_iou = smp.metrics.iou_score(
                    tp, fp, fn, tn,
                    reduction='macro'
                )
                val_iou_scores.append(batch_iou.item())

            avg_val_iou = np.mean(val_iou_scores)
            self._metric_values.append(avg_val_iou) # Store the average IoU for this epoch

            if avg_val_iou > self.best_metric:
                self.best_metric = avg_val_iou
                self.best_metric_epoch = epoch + 1
                self._save_model(epoch, avg_val_iou)
            else:
                self.scheduler.step()
                print('The learning rate updated to :', self.scheduler.get_last_lr()[0])
            print(f"current epoch: {epoch + 1} current IoU: {avg_val_iou:.4f}"
                  f" best IoU: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
