import os
import argparse

from omegaconf import OmegaConf
from pathlib import Path

from src.data_loader import DataGenerator
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
    train_dir = os.path.join('data', 'train', 'train')
    val_dir = os.path.join('data', 'valid', 'valid')

    train_loader = DataGenerator(train_dir,
                                 phase="train",
                                 batch_size=batch_size,
                                 shuffle=True).load_data()
    val_loader = DataGenerator(val_dir,
                               phase="val",
                               batch_size=len(os.listdir(val_dir)),
                               shuffle=False).load_data()

    trainer = Trainer(
        config=config,
        train_generator=train_loader,
        val_generator=val_loader
    )

    trainer.fit()


if __name__ == '__main__':
    main(args.config)
