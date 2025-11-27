"""
Detection services for video analysis.

This package contains detectors for faces, emotions, and activities.
"""

from src.services.detectors.face_detector import FaceDetector

# TODO: Implement EmotionAnalyzer and ActivityDetector in next phases
# from src.services.detectors.emotion_analyzer import EmotionAnalyzer
# from src.services.detectors.activity_detector import ActivityDetector

__all__ = ["FaceDetector"]
