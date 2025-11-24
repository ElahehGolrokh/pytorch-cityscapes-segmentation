import os
import argparse

from omegaconf import OmegaConf
from pathlib import Path

from src.data_loader import DataGenerator
from src.evaluating import Evaluator


parser = argparse.ArgumentParser(description="Train a segmentation model")
parser.add_argument("--config",
                    type=str,
                    default="config.yaml",
                    help="Path to the config file")
parser.add_argument("--output_name",
                    type=str,
                    default="evaluation_metrics.txt",
                    help="Name of the output file for evaluation metrics")
parser.add_argument("-s",
                    "--save_flag",
                    action='store_true',  # Default value is False
                    help='specifies whether to save evaluation metrics')
args = parser.parse_args()


def main(config_path,
         output_name,
         save_flag):
    config = OmegaConf.load(config_path)

    val_dir = os.path.join('data', 'valid', 'valid')
    val_loader = DataGenerator(val_dir,
                               phase="val",
                               batch_size=len(os.listdir(val_dir)),
                               shuffle=False).load_data()

    evaluator = Evaluator(
        config=config,
        model_path=Path("runs/best_model.pth"),
        val_loader=val_loader,
        output_name=output_name,
        save_flag=save_flag
    )

    evaluator.run()


if __name__ == '__main__':
    main(args.config,
         args.output_name,
         args.save_flag)
