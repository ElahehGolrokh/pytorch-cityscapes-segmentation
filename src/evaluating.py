import segmentation_models_pytorch as smp
import torch

from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader

from .utils import load_model, set_device


class Evaluator:
    """
    Evaluator for semantic segmentation models.

    Parameters:
    -----------
        config (OmegaConf): The configuration object.
        model_path (Path): The path to the trained model.
        val_loader (DataLoader): The validation data loader.
        save_flag (bool): Whether to save the evaluation results.
        output_name (str): The name of the output report.
        reduction (list): A list of the reduction modes for the evaluation metrics.
        num_classes (int): The number of classes in the segmentation task.

    Attributes:
    -----------
        device_ (torch.device | None): The device on which the model is loaded.
        model_ (torch.nn.Module | None): The loaded segmentation model.

    Private_Methods:
    ----------------
        _evaluate: Evaluate the model on the validation set.
        _calculate_metrics: Calculate evaluation metrics from the confusion matrix.
        _save_metrics: Save the evaluation metrics to a file.

    Public_Methods:
    ----------------
        run: Run the evaluation process.

    Example:
    ---------
    >>> evaluator = Evaluator(config, model_path, val_loader)
    >>> evaluator.run()
    """
    def __init__(self,
                 config: OmegaConf,
                 model_path: Path,
                 val_loader: DataLoader,
                 output_name: str,
                 save_flag: bool = True):
        self.config = config
        self.model_path = model_path
        self.val_loader = val_loader
        self.save_flag = save_flag
        self.output_name = output_name

        # Parameters from config
        self.reduction = self.config.evaluation.reduction_mode
        self.num_classes = len(self.config.dataset.classes)

        # placeholders
        self.device_: torch.device | None = None
        self.model_: torch.nn.Module | None = None

    def run(self, *args) -> None:
        """
        Run the evaluation process.

        args: Additional hyperparameters from the config which can be logged
        in the saved report.
        """
        tp, fp, fn, tn = self._evaluate()
        metrics = self._calculate_metrics(tp, fp, fn, tn)
        if self.save_flag:
            self._save_metrics(metrics, *args)

    def _evaluate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the model on the validation set.
        """
        # Setting device
        self.device_ = set_device()
        self.model_ = load_model(self.model_path, self.config, self.device_)
        self.model_.eval()
        # Lists to store metrics for each batch
        tp_list, fp_list, fn_list, tn_list = [], [], [], []
        with torch.no_grad():
            for val_data_batch in self.val_loader:
                val_images, val_labels = val_data_batch[0].to(self.device_).float(), val_data_batch[1].to(self.device_)
                val_outputs = self.model_(val_images)  # Logits (Batch, Classes, H, W)

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
                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
                tn_list.append(tn)

        # Aggregate statistics across all batches
        tp = torch.cat(tp_list)
        fp = torch.cat(fp_list)
        fn = torch.cat(fn_list)
        tn = torch.cat(tn_list)
        return tp, fp, fn, tn

    def _calculate_metrics(self,
                           tp: torch.Tensor,
                           fp: torch.Tensor,
                           fn: torch.Tensor,
                           tn: torch.Tensor) -> dict:
        """
        Calculate metrics using the aggregated statistics
        The `reduction` parameter applies to how the IoU is averaged across
        classes and/or images.
        'micro' means sum TP, FP, FN across classes and then calculate IoU.
        While 'macro' means calculate IoU for each class separately and then average.
        Use None to get per-class metrics.
        """
        metrics = {}
        for mode in self.reduction:
            print(f'mode = {mode}')
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction=mode)
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=mode)
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction=mode)
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction=mode)
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction=mode)

            if mode == "none":
                # reduction=None will get per-class metrics
                class_names = list(self.config.dataset.classes.keys())
                # reduction='none' returns shape [batch, classes]
                per_class_iou = iou_score.mean(dim=0)
                per_class_f1 = f1_score.mean(dim=0)
                per_class_precision = precision.mean(dim=0)
                per_class_recall = recall.mean(dim=0)
                per_class_accuracy = accuracy.mean(dim=0)

                for idx, cls in enumerate(class_names):
                    metrics[cls] = {
                        "iou_score": per_class_iou[idx].item(),
                        "f1_score": per_class_f1[idx].item(),
                        "precision": per_class_precision[idx].item(),
                        "recall": per_class_recall[idx].item(),
                        "accuracy": per_class_accuracy[idx].item(),
                    }
            else:
                # global (reduced) metrics
                metrics[f'{mode}_mean'] = {'iou_score': iou_score,
                                           'f1_score': f1_score,
                                           'precision': precision,
                                           'recall': recall,
                                           'accuracy': accuracy}
                print(f"\n--- Test Set Evaluation Metrics, reduction mode: {mode} ---")
                print(f"Mean IoU: {iou_score.item():.4f}")
                print(f"Mean F1-Score (Dice): {f1_score.item():.4f}")
                print(f"Mean Precision: {precision.item():.4f}")
                print(f"Mean Recall: {recall.item():.4f}")
                print(f"Mean Accuracy: {accuracy.item():.4f}")
        print("-----------------------------------")
        return metrics

    def _save_metrics(self, metrics: dict, *args) -> None:
        """
        Save evaluation metrics to a file
        """
        logs_dir = self.config.dirs.logs
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        metrics_file = Path(logs_dir) / self.output_name
        with open(metrics_file, "w") as f:
            f.write("---------------EVALUATION METRICS---------------\n\n")
            # Write config parameters passed as *args
            # Only print configs if args contains at least one non-empty dict/item
            if any((isinstance(item, dict) and item) or (not isinstance(item, dict)) for item in args):
                f.write("CONFIGS:\n")
                for item in args:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            f.write(f"  {k.split('.')[-1]}: {v}\n")
                    else:
                        f.write(f"  {item}\n")
                f.write("\n")
                f.write("-------------------Results-------------------\n")
            for key, value in metrics.items():
                f.write("---------------------------------------------\n")
                f.write(f"Class = {key}\n")
                for nest_key, nest_value in value.items():
                    print(f"nests prints_{nest_key}: {nest_value:.4f}")
                    f.write(f"{nest_key}: {nest_value:.4f}\n")
            f.write("---------------------------------------------\n")
        print(f"Metrics saved to {metrics_file}")
