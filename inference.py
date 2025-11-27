import argparse
import os
import pandas as pd

from omegaconf import OmegaConf
from pathlib import Path

from src.data_loader import DataGenerator, create_data_paths
from src.prediction import SceneSegmentor


parser = argparse.ArgumentParser(description="Inference for a segmentation model")
parser.add_argument("--config",
                    type=str,
                    default="config.yaml",
                    help="Path to the config file")
args = parser.parse_args()


def main(config_path):
    config = OmegaConf.load(config_path)

    test_dir = os.path.join('data', 'test', 'image')
    test_df = pd.read_csv(os.path.join('data', 'test.csv'))
    paths = create_data_paths(test_dir, test_df)
    test_loader = DataGenerator(config=config,
                                phase="test",
                                batch_size=len(paths),
                                shuffle=False).load_data(paths)
    model_path=Path("runs/best_model_epoch76_0.6119.pth")
    segmentor = SceneSegmentor(
        config=config,
        model_path=model_path,
        test_generator=test_loader,
    )
    segmentor.run()


if __name__ == '__main__':
    main(args.config)
