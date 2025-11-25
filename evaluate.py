import os
import argparse
import logging

from omegaconf import OmegaConf
from pathlib import Path

from src.data_loader import DataGenerator, create_data_paths
from src.evaluating import Evaluator
from src.utils import get_config_value


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
parser.add_argument(
    "--report",
    nargs="*",
    default=[],
    help="List of config keys to add to the saved report. Example: training.batch_size training.lr"
)  # will give you: args.report == ["training.batch_size", "training.lr", ...]
parser_args = parser.parse_args()


def main(config_path,
         output_name,
         save_flag,
         *args):
    config = OmegaConf.load(config_path)
    if args and not save_flag:
        logging.warning("You passed report items but the save_flag is not set."
                        " Evaluation report will not be saved.")
    if args:
        report_items = {}
        for key in args:
            report_items[key] = get_config_value(config, key)
    else:
        report_items = {}
    val_dir = os.path.join('data', 'valid', 'valid')
    paths = create_data_paths(val_dir)
    val_loader = DataGenerator(phase="val",
                               batch_size=len(paths),
                               shuffle=False).load_data(paths)

    evaluator = Evaluator(
        config=config,
        model_path=Path("runs/best_model.pth"),
        val_loader=val_loader,
        output_name=output_name,
        save_flag=save_flag
    )
    evaluator.run(report_items)


if __name__ == '__main__':
    main(parser_args.config,
         parser_args.output_name,
         parser_args.save_flag,
         *parser_args.report)
