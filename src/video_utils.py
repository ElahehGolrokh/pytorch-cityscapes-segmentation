import cv2
import numpy as np

from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from .utils import mask_to_rgb


class VideoProcessor:
    def __init__(self,
                 video_path: Path):
        self.video_path = video_path

    def _load_video(self) -> cv2.VideoCapture:
        """
        Loads a video file using OpenCV.
        Args:
            video_path (str): The path to the video file.
        Returns:
            cv2.VideoCapture: The video capture object.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

    def get_video_frames(self):
        try:
            self._load_video()
            video_frames = []
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process the frame
                np_frame = np.array(frame)/255
                video_frames.append(np_frame)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f'frames length: {len(video_frames)}')
            return fps, video_frames
        except Exception as e:
            raise ValueError(f"Error processing video {self.video_path}: {e}") from e
        finally:
            self.cap.release()


class VideoWriter:
    def __init__(self,
                 config: OmegaConf,
                 fps: float,
                 output_path: str):
        self.output_path = output_path
        self.writer = None
        self.fps = fps

        # Parameters from config
        self.color_map = config.dataset.color_map
        self.frame_width = config.dataset.width
        self.frame_height = config.dataset.height
    
    def run(self, masks) -> None:
        self._open()
        self._write_masks(masks)
        self._release()
    
    def _open(self):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
    
    def _write_masks(self, masks):
        """Write batch of masks as RGB frames"""
        print('writing masks...')
        for mask in tqdm(masks):
            rgb_frame = mask_to_rgb(np.array(mask, dtype=np.uint8), self.color_map)
            self.writer.write(rgb_frame)
    
    def _release(self):
        if self.writer:
            self.writer.release()
