import numpy as np
import segmentation_models_pytorch as smp
import torch


class Trainer:
    def __init__(self,
                 train_generator: torch.utils.data.DataLoader,
                 val_generator: torch.utils.data.DataLoader,
                 num_classes: int,
                 model: torch.nn.Module,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.Module,
                 scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.num_classes = num_classes
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self._metric_values = []   # This will store average IoU for each epoch
        self._epoch_loss_values = []

    def fit(self,
            epochs: int,
            ):
        # Initialize with a value that will be easily surpassed
        self.best_metric = -1
        self.best_metric_epoch = -1
        for epoch in range(epochs):
            print('-' * 10)
            print(f"epoch {epoch + 1}/{epochs}")
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
            self.scheduler.step()
            # Use get_last_lr() for PyTorch 1.4+ and later versions
            print('\n', f"###### epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            print('The learning rate updated to :', self.scheduler.get_last_lr()[0])
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
                    reduction='micro'
                )
                val_iou_scores.append(batch_iou.item())

            avg_val_iou = np.mean(val_iou_scores)
            self._metric_values.append(avg_val_iou) # Store the average IoU for this epoch

            if avg_val_iou > self.best_metric:
                self.best_metric = avg_val_iou
                self.best_metric_epoch = epoch + 1
                torch.save(self.model.state_dict(), f'best_model_epoch{epoch}_{avg_val_iou:.4f}.pth')
                print('###### Congratulations ###### saved new best metric model')
            print(f"current epoch: {epoch + 1} current IoU: {avg_val_iou:.4f}"
                    f" best IoU: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
