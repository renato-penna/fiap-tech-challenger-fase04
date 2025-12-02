import cv2
import numpy as np
from typing import List, Generator, Tuple
import logging
from src.models.scene import Scene

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Detects scenes in a video using histogram difference.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the SceneDetector.
        
        Args:
            threshold: Threshold for histogram difference (0.0 to 1.0).
                      Higher values mean less sensitivity (fewer scenes).
                      Lower values mean more sensitivity (more scenes).
                      Default 0.5 is a good starting point for correlation.
        """
        self.threshold = threshold
        
    def detect_scenes(self, video_path: str) -> List[Scene]:
        """
        Detect scenes in the video file.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            List of Scene objects.
        """
        if not video_path:
            raise ValueError("Video path cannot be empty")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        scene_start_frame = 0
        scene_id = 1
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return []
            
        # Convert to HSV for better color comparison (less sensitive to lighting)
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for first frame
        # Using 50 bins for Hue, 60 for Saturation
        prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        
        frame_num = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to HSV
            curr_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram
            curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms using Correlation method
            # 1.0 = identical, 0.0 = completely different
            score = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            
            # If correlation is low, it's a scene change
            # We use (1 - score) > threshold logic, but since score is correlation:
            # If score < threshold, it means they are different enough
            if score < self.threshold:
                # Scene change detected
                scene_end_frame = frame_num - 1
                
                # Create scene object
                scene = Scene(
                    scene_id=scene_id,
                    start_frame=scene_start_frame,
                    end_frame=scene_end_frame,
                    start_time=scene_start_frame / fps,
                    end_time=scene_end_frame / fps
                )
                scenes.append(scene)
                
                # Start new scene
                scene_id += 1
                scene_start_frame = frame_num
                
            prev_hist = curr_hist
            frame_num += 1
            
        # Add the last scene
        scene = Scene(
            scene_id=scene_id,
            start_frame=scene_start_frame,
            end_frame=frame_num - 1,
            start_time=scene_start_frame / fps,
            end_time=(frame_num - 1) / fps
        )
        scenes.append(scene)
        
        cap.release()
        return scenes
