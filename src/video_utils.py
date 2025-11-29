import cv2
import numpy as np

from pathlib import Path


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
        self._load_video()
        video_frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process the frame
            np_frame = np.array(frame)/255
            video_frames.append(np_frame)
            
        print(f'frames length: {len(video_frames)}')
        return video_frames
