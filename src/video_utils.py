import cv2
import numpy as np

from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from .memory_utils import MemoryMonitor
from .utils import mask_to_rgb


class VideoProcessor:
    def __init__(self,
                 video_path: Path,
                 memory_threshold: float = 80.0,
                 check_interval: int = 10):
        self.video_path = video_path
        self.memory_monitor = MemoryMonitor(threshold=memory_threshold,
                                            check_interval=check_interval)

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

            # Check memory before starting
            self.memory_monitor.check(force=True)
            
            frame_count = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process the frame
                np_frame = np.array(frame)/255
                video_frames.append(np_frame)
                frame_count += 1

                # ✅ Memory check - will raise MemoryError if threshold exceeded
                self.memory_monitor.check()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Final memory check
            mem_info = self.memory_monitor.get_current_usage()
            print(f"  Final memory usage: {mem_info['percent']:.1f}% ({mem_info['used_gb']:.1f}GB)\n")
            print(f'frames length: {len(video_frames)}')
            return fps, video_frames
        except MemoryError as e:
            # Memory limit exceeded - clean up and re-raise
            print(f"\n❌ Memory limit exceeded after loading {frame_count} frames")
            print(f"   {str(e)}")
            print(f"   Loaded frames so far: {len(video_frames)}")
            print(f"   Consider: reducing video resolution or processing in chunks\n")
            raise  # Re-raise to stop execution
        except Exception as e:
            raise ValueError(f"Error processing video {self.video_path}: {e}") from e
        finally:
            self.cap.release()
            # Final memory check
            mem_info = self.memory_monitor.get_current_usage()
            print(f"  Memory usage: {mem_info['percent']:.1f}% ({mem_info['used_gb']:.1f}GB)")


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
