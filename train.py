import argparse
import os
import pandas as pd

from omegaconf import OmegaConf

from src.data_loader import DataGenerator, create_data_paths
from src.training import Trainer


parser = argparse.ArgumentParser(description="Train a segmentation model")
parser.add_argument("--config",
                    type=str,
                    default="config.yaml",
                    help="Path to the config file")
args = parser.parse_args()


def main(config_path):
    config = OmegaConf.load(config_path)
    batch_size = config.training.batch_size

    train_dir = os.path.join('data', 'train', 'image')
    val_dir = os.path.join('data', 'val', 'image')

    train_df = pd.read_csv(os.path.join('data', 'train.csv'))
    val_df = pd.read_csv(os.path.join('data', 'val.csv'))

    train_paths = create_data_paths(train_dir, train_df)
    val_paths = create_data_paths(val_dir, val_df)

    train_loader = DataGenerator(config=config,
                                 phase="train",
                                 batch_size=batch_size,
                                 shuffle=True).load_data(train_paths)
    val_loader = DataGenerator(config=config,
                               phase="val",
                               batch_size=len(val_paths),
                               shuffle=False).load_data(val_paths)

    trainer = Trainer(
        config=config,
        train_generator=train_loader,
        val_generator=val_loader
    )

    trainer.fit()


if __name__ == '__main__':
    main(args.config)
