import cv2
import os
from typing import Generator, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes videos frame by frame using OpenCV.

    This class allows opening a video file, reading frames individually,
    and retrieving video information (FPS, duration, resolution).
    """

    def __init__(self, video_path: str):
        """
        Initialize the video processor.

        Args:
            video_path: Path to the video file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the video cannot be opened
        """
        # Validate file existence
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        # Open video
        self.cap = cv2.VideoCapture(video_path)

        # Verify if opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video loaded: {self.width}x{self.height} @ "
            f"{self.fps}fps, {self.frame_count} frames"
        )

    def get_frames(
        self
    ) -> Generator[Tuple[int, float, cv2.Mat], None, None]:
        """
        Generates video frames one by one.

        Yields:
            Tuple[int, float, cv2.Mat]: (frame_number, timestamp, frame)
                - frame_number: frame number (0, 1, 2, ...)
                - timestamp: time in seconds since start
                - frame: frame image (numpy array BGR)

        Example:
            >>> processor = VideoProcessor("video.mp4")
            >>> for frame_num, timestamp, frame in processor.get_frames():
            ...     print(f"Frame {frame_num} at {timestamp:.2f}s")
        """
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # End of video

            timestamp = frame_num / self.fps if self.fps > 0 else 0
            yield frame_num, timestamp, frame
            frame_num += 1

    def release(self):
        """Release video resources."""
        self.cap.release()

    def get_video_info(self) -> dict:
        """
        Return video information.

        Returns:
            dict: Dictionary with video information:
                - fps: frames per second
                - frame_count: total number of frames
                - width: width in pixels
                - height: height in pixels
                - duration: duration in seconds
        """
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": (
                self.frame_count / self.fps if self.fps > 0 else 0
            )
        }
