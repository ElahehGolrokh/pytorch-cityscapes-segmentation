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
parser.add_argument("-ip",
                    "--image-path",
                    type=str,
                    default=None,
                    help="Path to the input image file or the directory containing images")
args = parser.parse_args()


def main(config_path: Path,
         image_path: Path):
    config = OmegaConf.load(config_path)

    if image_path is None:
        test_dir = os.path.join('data', 'test', 'image')
        test_df = pd.read_csv(os.path.join('data', 'test.csv'))
        paths = create_data_paths(test_dir, test_df)
    elif os.path.isdir(image_path):
        paths = create_data_paths(image_path)
    else:
        paths = [image_path]

    test_loader = DataGenerator(config=config,
                                phase="test",
                                batch_size=len(paths),
                                shuffle=False).load_data(paths)
    model_path = Path("runs/best_model_epoch76_0.6119_efficientnetb3.pth")
    segmentor = SceneSegmentor(
        config=config,
        model_path=model_path,
        test_generator=test_loader,
    )
    segmentor.run()


if __name__ == '__main__':
    main(args.config,
         args.image_path)
