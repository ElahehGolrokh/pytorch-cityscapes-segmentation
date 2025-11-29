import argparse
import os
import pandas as pd

from omegaconf import OmegaConf
from pathlib import Path

from src.data_loader import DataGenerator, create_data_paths
from src.prediction import SceneSegmentor, VideoProcessor
from src.utils import create_video_output
from src.visualization import SegmentationVisualizer


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
parser.add_argument("-vp",
                    "--video-path",
                    type=str,
                    default=None,
                    help="Path to the input video file")
parser.add_argument("-n",
                    "--number_of_visualizations",
                    type=int,
                    default=None,
                    help="Number of visualizations to generate")
args = parser.parse_args()


def main(config_path: Path,
         image_path: Path,
         video_path: Path,
         number_of_visualizations: int):
    config = OmegaConf.load(config_path)
    model_path = Path("runs/best_model_epoch76_0.6119_efficientnetb3.pth")

    if video_path is None:
        if image_path is None:
            # If no image path is provided, use the test dataset
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

        segmentor = SceneSegmentor(
            config=config,
            model_path=model_path,
            test_generator=test_loader,
        )
        predictions = segmentor.run()
        SegmentationVisualizer(config=config,
                               images_path=paths,
                               pr_masks=predictions,
                               to_visualize_samples=number_of_visualizations).visualize()
    else:
        # Video inference
        video_processor = VideoProcessor(video_path=video_path)
        video_frames = video_processor.get_video_frames()
        test_loader = DataGenerator(config=config,
                                    phase="test",
                                    batch_size=len(video_frames),
                                    shuffle=False).load_data(video_frame=video_frames)

        segmentor = SceneSegmentor(
                config=config,
                model_path=model_path,
                test_generator=test_loader,
            )
        predicted_masks = segmentor.run()
        create_video_output(video_processor.cap,
                            config=config,
                            predicted_masks=predicted_masks,
                            output_path="output_video.avi")


if __name__ == '__main__':
    main(args.config,
         args.image_path,
         args.video_path,
         args.number_of_visualizations)
