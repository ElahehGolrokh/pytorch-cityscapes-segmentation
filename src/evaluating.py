import segmentation_models_pytorch as smp
import torch

from omegaconf import OmegaConf
from pathlib import Path


class Evaluator:
    def __init__(self,
                 config: OmegaConf,
                 model_path: Path,
                 val_loader,
                 output_name: str,
                 save_flag: bool = True):
        self.config = config
        self.model_path = model_path
        self.val_loader = val_loader
        self.save_flag = save_flag
        self.output_name = output_name

        self.reduction = self.config.evaluation.reduction_mode
        self.num_classes = len(self.config.dataset.classes)
    
    def run(self, *args):
        tp, fp, fn, tn = self._evaluate()
        metrics = self._claculate_metrics(tp, fp, fn, tn)
        if self.save_flag:
            self._save_metrics(metrics, *args)
    
    def _load_model(self):
        # Re-instantiate the model with the correct architecture
        # Setting device
        # try:
        #     device = torch.device("cuda:0")
        #     print('run with gpu')
        # except:
        self.device = torch.device("cpu")
        print(f"Using {self.device} device")

        encoder_name = self.config.training.encoder_name
        activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
        self.model = smp.Unet(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,                     # model output classes (number of classes in your dataset)
            activation=activation
        ).to(self.device)

        # Load the saved state dictionary
        try:
            self.model.load_state_dict(torch.load(self.model_path,
                                                  map_location=self.device))
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model state_dict: {e}\n"
                             f"Ensure the path is correct and the model architecture matches.")

    def _evaluate(self):
        self._load_model()
        self.model.eval()
        # Lists to store metrics for each batch
        tp_list, fp_list, fn_list, tn_list = [], [], [], []
        with torch.no_grad():
            for val_data_batch in self.val_loader:
                val_images, val_labels = val_data_batch[0].to(self.device).float(), val_data_batch[1].to(self.device)
                val_outputs = self.model(val_images)  # Logits (Batch, Classes, H, W)

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
    
    def _claculate_metrics(self, tp, fp, fn, tn):
        # Calculate metrics using the aggregated statistics
        # The `reduction` parameter applies to how the IoU is averaged across classes and/or images.
        # 'micro' means sum TP, FP, FN across classes and then calculate IoU.
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction=self.reduction)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=self.reduction) # Also known as Dice Coefficient
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction=self.reduction)
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction=self.reduction)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction=self.reduction)

        print(f"\n--- Test Set Evaluation Metrics, reduction mode: {self.reduction} ---")
        print(f"Mean IoU: {iou_score.item():.4f}")
        print(f"Mean F1-Score (Dice): {f1_score.item():.4f}")
        print(f"Mean Precision: {precision.item():.4f}")
        print(f"Mean Recall: {recall.item():.4f}")
        print(f"Mean Accuracy: {accuracy.item():.4f}")
        print("-----------------------------------")
        return {'iou_score': iou_score,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy}

    def _save_metrics(self, metrics: dict, *args):
        logs_dir = self.config.dirs.logs
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        metrics_file = Path(logs_dir) / self.output_name
        with open(metrics_file, "w") as f:
            f.write(f"---------------Evaluation Metrics for {self.reduction} mode---------------\n\n")
            # Write config parameters passed as *args
            # Only print configs if args contains at least one non-empty dict/item
            if any((isinstance(item, dict) and item) or (not isinstance(item, dict)) for item in args):
                f.write("Configs:\n")
                for item in args:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            f.write(f"  {k.split('.')[-1]}: {v}\n")
                    else:
                        f.write(f"  {item}\n")
                f.write("\n")
                f.write("---------------Results---------------\n")

            for key, value in metrics.items():
                f.write(f"{key}: {value.item():.4f}\n")
        print(f"Metrics saved to {metrics_file}")
