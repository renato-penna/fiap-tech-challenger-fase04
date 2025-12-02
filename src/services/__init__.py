"""
Services for video processing and analysis.

This package contains services for video processing, face detection,
emotion analysis, and activity detection.
"""

from src.services.video_processor import VideoProcessor
from src.services.detectors import FaceDetector
from src.services.detectors import EmotionAnalyzer
from src.services.detectors import ActivityDetector

__all__ = [
    'VideoProcessor',
    'FaceDetector',
    'EmotionAnalyzer',
    'ActivityDetector'
]
