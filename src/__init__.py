"""
FIAP Tech Challenge Phase 4 - Video Analysis System

This package provides a complete video analysis system that:
1. Reads video frames (VideoProcessor)
2. Detects faces using YOLO Pose (FaceDetector)
3. Analyzes emotions (EmotionAnalyzer) - TODO
4. Detects activities (ActivityDetector) - TODO
5. Generates reports (ReportGenerator) - TODO

Main components:
- models: Data classes for bounding boxes, detections, and analysis results
- services: Core services for video processing and detection
- utils: Utility functions for visualization and reporting
"""

from src.models import (
    BoundingBox,
    FaceDetection,
    EmotionAnalysis,
    ActivityDetection
)

from src.services import (
    VideoProcessor,
    FaceDetector
)

from src.utils import (
    draw_bounding_box,
    draw_face_detections,
    add_frame_info,
    frame_to_rgb,
    process_video_with_detections
)

__version__ = "0.1.0"

__all__ = [
    # Models
    'BoundingBox',
    'FaceDetection',
    'EmotionAnalysis',  # Model exists, analyzer TODO
    'ActivityDetection',  # Model exists, detector TODO
    # Services
    'VideoProcessor',
    'FaceDetector',
    # Utils
    'draw_bounding_box',
    'draw_face_detections',
    'add_frame_info',
    'frame_to_rgb',
    'process_video_with_detections',
]

